#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#GPI-CBU algorithm

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Literal, Tuple, Dict, Any, Optional

import argparse
import json
import random
import time
from collections import deque

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# -----------------------------
# Config / hyperparameters
# -----------------------------

@dataclass(frozen=True)
class PhysicsParams:
    """Problem parameters."""
    T: float = 1.0              # Terminal time
    lamb: float = 0.25          # Terminal condition coefficient
    theta: float = 1.0          # Running cost coefficient
    beta: float = 2.0           # Jump term coefficient

    dim_x: int = 10            # State dimension
    dim_w: int = 10             # Brownian dimension

    # Jump distribution (Gaussian)
    mu_j: float = 0.0           # Mean jump size (scalar, expanded to vector)
    cov_j_max: float = 0.4      # Diagonal jump variance upper bound

    # Diffusion matrix sampling
    sigma_scale: float = 0.3    # std of entries in sigma (random init)


@dataclass(frozen=True)
class ModelParams:
    """Neural-net architecture parameters."""
    kind: Literal["dgm", "mlp"] = "dgm"
    hidden_dim: int = 150
    n_layers: int = 3
    mlp_width: int = 128        # used if kind="mlp"
    use_batchnorm: bool = True  # used if kind="mlp"


@dataclass(frozen=True)
class TrainPhase:
    """One training phase."""
    name: str
    epochs: int
    lr_value: float
    lr_control: float
    optimizer: Literal["adam", "adagrad"] = "adam"
    grad_clip_norm: float = 50.0
    target_tau: float = 0.0     # EMA on value target: new = tau*target + (1-tau)*online
    log_every: int = 10


@dataclass(frozen=True)
class TrainParams:
    # Sampling
    num_samples: int = 2048     # interior samples per epoch
    nc_samples: int = 2048      # terminal samples per epoch

    # Replay buffer
    buffer_mult: int = 64       # buffer_size = num_samples * buffer_mult

    # Batching
    batch_size: int = 1024

    # Test / diagnostics
    test_samples: int = 1000
    bc_weight: float = 45.0     # weight on terminal condition loss


@dataclass(frozen=True)
class RunParams:
    seed: int = 22
    use_mirrored_strategy: bool = True
    save_weights: bool = True
    weights_prefix: str = "1lqrw"
    make_plots: bool = True
    plot_prefix: str = "LQR"
    device_log: bool = False


@dataclass(frozen=True)
class Config:
    physics: PhysicsParams = PhysicsParams()
    model: ModelParams = ModelParams()
    train: TrainParams = TrainParams()
    phases: Tuple[TrainPhase, ...] = (
        TrainPhase(name="warmup_adam", epochs=100, lr_value=2e-4 / 5, lr_control=2e-4, optimizer="adam"),
        TrainPhase(name="finetune_adagrad", epochs=100, lr_value=5e-3, lr_control=5e-3, optimizer="adagrad"),
    )
    run: RunParams = RunParams()


def _load_config_file(path: str) -> Dict[str, Any]:
    """Load TOML (preferred) or JSON config."""
    if path.lower().endswith(".toml"):
        import tomllib  # py3.11+
        with open(path, "rb") as f:
            return tomllib.load(f)
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError("Config must be .toml or .json")


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict update."""
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def config_from_file(path: str) -> Config:
    raw = _load_config_file(path)

    # Start from defaults, then override.
    default = asdict(Config())
    merged = _deep_update(default, raw)

    # Rebuild dataclasses
    physics = PhysicsParams(**merged["physics"])
    model = ModelParams(**merged["model"])
    train = TrainParams(**merged["train"])
    phases = tuple(TrainPhase(**p) for p in merged["phases"])
    run = RunParams(**merged["run"])
    return Config(physics=physics, model=model, train=train, phases=phases, run=run)


# -----------------------------
# Utilities
# -----------------------------

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau: float) -> None:
    """EMA update: target <- tau*target + (1-tau)*source."""
    if tau <= 0.0:
        target.set_weights(source.get_weights())
        return
    tw = target.get_weights()
    sw = source.get_weights()
    target.set_weights([tau * t + (1.0 - tau) * s for t, s in zip(tw, sw)])


# -----------------------------
# Replay Buffer
# -----------------------------

class ReplayBuffer:
    """Simple python replay buffer storing tuples of numpy arrays."""

    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def add_batch(
        self,
        x: np.ndarray,
        t: np.ndarray,
        e: np.ndarray,
        xbcT: np.ndarray,
        tbcT: np.ndarray,
        VbcT: np.ndarray,
    ) -> None:
        for i in range(x.shape[0]):
            self._buf.append((x[i], t[i], e[i], xbcT[i], tbcT[i], VbcT[i]))

    def size(self) -> int:
        return len(self._buf)

    def as_arrays(self) -> Tuple[np.ndarray, ...]:
        x, t, e, xbcT, tbcT, VbcT = zip(*self._buf)
        return (
            np.asarray(x),
            np.asarray(t),
            np.asarray(e),
            np.asarray(xbcT),
            np.asarray(tbcT),
            np.asarray(VbcT),
        )


# -----------------------------
# Reference (analytic / numerical) solution helpers
# -----------------------------

class LQRReference:
    """
    Cache H(t) by solving ODE once and building an interpolant.
    Also cache integral_{t}^{T} H(s) ds via precomputed trapezoid rule.

    This replaces opt_H/value1/control1 and is much faster for repeated calls.
    """

    def __init__(self, physics: PhysicsParams, cov_j: np.ndarray):
        self.p = physics
        self.tr_cov_j = float(np.trace(cov_j))

        def dH_dt(t: float, H: np.ndarray) -> np.ndarray:
            H0 = H[0]
            if H0 < 0:
                return np.array([0.0], dtype=np.float64)
            num = 2 * self.p.theta * H0**2 + self.p.beta * self.tr_cov_j * H0**3
            den = (2 * self.p.theta + self.p.beta * self.tr_cov_j * H0) ** 2
            return np.array([num / den], dtype=np.float64)

        H_T = np.array([2.0 * self.p.lamb], dtype=np.float64)
        sol = solve_ivp(
            dH_dt,
            (self.p.T, 0.0),
            H_T,
            method="RK45",
            t_eval=np.linspace(self.p.T, 0.0, 2000),
            dense_output=True,
        )

        # Dense solution
        self._H_dense = sol.sol

        # Precompute integral I(t) = ∫_t^T H(s) ds on an increasing grid.
        grid = np.linspace(0.0, self.p.T, 2000)
        H_grid = self.H(grid)
        # cumulative trapezoid from 0 to t
        cum = np.concatenate([[0.0], np.cumsum((H_grid[1:] + H_grid[:-1]) * np.diff(grid) / 2.0)])
        I_grid = cum[-1] - cum  # ∫_t^T H(s) ds
        self._I_interp = interp1d(grid, I_grid, kind="cubic", fill_value="extrapolate", assume_sorted=True)

    def H(self, t: np.ndarray | float) -> np.ndarray:
        t_arr = np.asarray(t, dtype=np.float64)
        # The dense solution was built on reversed time; sol(t) still works for [0,T].
        return np.asarray(self._H_dense(t_arr)[0], dtype=np.float64)

    def I(self, t: np.ndarray | float) -> np.ndarray:
        return np.asarray(self._I_interp(np.asarray(t, dtype=np.float64)), dtype=np.float64)

    def value(self, t: float, x: np.ndarray, sigma: np.ndarray) -> float:
        Ht = float(self.H(t))
        It = float(self.I(t))
        trace_sig = float(np.trace(sigma @ sigma.T))
        return 0.5 * Ht * float(np.sum(x**2)) + 0.5 * It * trace_sig

    def control(self, t: float, x: np.ndarray) -> np.ndarray:
        Ht = float(self.H(t))
        denom = 2.0 * self.p.theta + self.tr_cov_j * self.p.beta * Ht
        return -Ht * x / denom


# -----------------------------
# Models
# -----------------------------

class DenseNet(tf.keras.Model):
    """Simple MLP with optional BatchNorm, used to replace BSPINN_* duplication."""

    def __init__(self, width: int, depth: int, output_dim: int, use_batchnorm: bool = True):
        super().__init__()
        self.layers_ = []
        for _ in range(depth):
            self.layers_.append(tf.keras.layers.Dense(width, activation="tanh"))
            if use_batchnorm:
                self.layers_.append(tf.keras.layers.BatchNormalization())
        self.out = tf.keras.layers.Dense(output_dim, activation=None)

    def call(self, inputs, training: bool = False):
        x, t = inputs
        xt = tf.concat([x, t], axis=1)
        h = xt
        for layer in self.layers_:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                h = layer(h, training=training)
            else:
                h = layer(h)
        return self.out(h)


class DGMNet(tf.keras.Model):
    """DGM network (shared implementation for value and control)."""

    def __init__(self, hidden_dim: int, n_layers: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n = n_layers

        self.sig_act = tf.keras.layers.Activation(tf.nn.tanh)

        self.Sw = tf.keras.layers.Dense(hidden_dim)
        self.Uz = tf.keras.layers.Dense(hidden_dim)
        self.Wsz = tf.keras.layers.Dense(hidden_dim)
        self.Ug = tf.keras.layers.Dense(hidden_dim)
        self.Wsg = tf.keras.layers.Dense(hidden_dim)
        self.Ur = tf.keras.layers.Dense(hidden_dim)
        self.Wsr = tf.keras.layers.Dense(hidden_dim)
        self.Uh = tf.keras.layers.Dense(hidden_dim)
        self.Wsh = tf.keras.layers.Dense(hidden_dim)
        self.Wf = tf.keras.layers.Dense(output_dim)

    def build(self, input_shape):
        x_shape, t_shape = input_shape
        xt_shape = (x_shape[0], x_shape[1] + t_shape[1])
        self.Sw.build(xt_shape)
        self.Uz.build(xt_shape)
        self.Wsz.build((None, self.hidden_dim))
        self.Ug.build(xt_shape)
        self.Wsg.build((None, self.hidden_dim))
        self.Ur.build(xt_shape)
        self.Wsr.build((None, self.hidden_dim))
        self.Uh.build(xt_shape)
        self.Wsh.build((None, self.hidden_dim))
        self.Wf.build((None, self.hidden_dim))
        super().build(input_shape)

    def call(self, inputs):
        x, t = inputs
        xt = tf.concat([x, t], axis=1)
        S1 = self.sig_act(self.Sw(xt))
        out = S1
        for _ in range(1, self.n):
            S = out
            Z = self.sig_act(self.Uz(xt) + self.Wsz(S))
            # paper typo: G should use S (not S1)
            G = self.sig_act(self.Ug(xt) + self.Wsg(S))
            R = self.sig_act(self.Ur(xt) + self.Wsr(S))
            H = self.Uh(xt) + self.Wsh(S * R)
            out = (1.0 - G) * H + Z * S
        return self.Wf(out)


def build_models(cfg: Config) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Returns (value_model, control_model, value_target_model)."""
    p = cfg.physics
    mp = cfg.model

    if mp.kind == "dgm":
        V = DGMNet(mp.hidden_dim, mp.n_layers, output_dim=1)
        U = DGMNet(mp.hidden_dim, mp.n_layers, output_dim=p.dim_x)
        Vt = DGMNet(mp.hidden_dim, mp.n_layers, output_dim=1)
    else:
        # depth fixed at 4 to match the original BSPINN_* roughly
        V = DenseNet(width=mp.mlp_width, depth=4, output_dim=1, use_batchnorm=mp.use_batchnorm)
        U = DenseNet(width=mp.mlp_width, depth=4, output_dim=p.dim_x, use_batchnorm=mp.use_batchnorm)
        Vt = DenseNet(width=mp.mlp_width, depth=4, output_dim=1, use_batchnorm=mp.use_batchnorm)

    return V, U, Vt


# -----------------------------
# PINN losses (refactored)
# -----------------------------

@tf.function(jit_compile=True)
def _fprime(
    h: tf.Tensor,
    V_model: tf.keras.Model,
    U_model: tf.keras.Model,
    x: tf.Tensor,
    t: tf.Tensor,
    sigma: tf.Tensor,
) -> tf.Tensor:
    """
    Vectorized variant of the original fprime:
    sum_i V(x + h^2 * u / (2*dim_w) + h/sqrt(2) * sigma[:,i], t + h^2/(2*dim_w))
    """
    batch = tf.shape(x)[0]
    dim_x = tf.shape(x)[1]
    dim_w = tf.shape(sigma)[1]

    t2 = t + (h ** 2) / (2.0 * tf.cast(dim_w, tf.float32))
    common = x + (h ** 2) * U_model([x, t]) / (2.0 * tf.cast(dim_w, tf.float32))

    # sigma_cols: (dim_w, 1, dim_x) then broadcast to (dim_w, batch, dim_x)
    sigma_cols = tf.transpose(sigma)[..., tf.newaxis]  # (dim_w, dim_x, 1)
    sigma_cols = tf.transpose(sigma_cols, perm=[0, 2, 1])  # (dim_w, 1, dim_x)
    sigma_cols = tf.broadcast_to(sigma_cols, (dim_w, batch, dim_x))

    h_scaled = h / tf.sqrt(tf.constant(2.0, dtype=tf.float32))  # (batch,1)
    h_scaled = tf.broadcast_to(h_scaled[tf.newaxis, :, :], (dim_w, batch, 1))

    x2 = common[tf.newaxis, :, :] + h_scaled * sigma_cols  # (dim_w, batch, dim_x)
    t2_rep = tf.broadcast_to(t2[tf.newaxis, :, :], (dim_w, batch, 1))

    x2_flat = tf.reshape(x2, (-1, dim_x))
    t2_flat = tf.reshape(t2_rep, (-1, 1))

    V_out = V_model([x2_flat, t2_flat])
    V_out = tf.reshape(V_out, (dim_w, batch, -1))
    return tf.reduce_sum(V_out, axis=0)  # (batch, 1)


def pinn_loss_value(
    V_model: tf.keras.Model,
    U_model: tf.keras.Model,
    V_target: tf.keras.Model,
    x: tf.Tensor,
    t: tf.Tensor,
    e: tf.Tensor,
    xbcT: tf.Tensor,
    tbcT: tf.Tensor,
    VbcT: tf.Tensor,
    physics: PhysicsParams,
    sigma: tf.Tensor,
    bc_weight: float,
) -> tf.Tensor:
    """
    Original pinn_loss1bis, but:
    - no global variables
    - h is constructed as a zero tensor and watched for derivatives
    """
    x = tf.cast(x, tf.float32)
    t = tf.cast(t, tf.float32)
    e = tf.cast(e, tf.float32)

    # Second derivative wrt h at h=0
    h = tf.zeros((tf.shape(x)[0], 1), dtype=tf.float32)

    with tf.GradientTape() as tape2:
        tape2.watch(h)
        with tf.GradientTape() as tape1:
            tape1.watch(h)
            f_val = _fprime(h, V_target, U_model, x, t, sigma)  # (batch,1)
        df_dh = tape1.gradient(f_val, h)
    d2f_dh2 = tape2.gradient(df_dh, h)  # (batch,1)

    u = U_model([x, t])  # (batch, dim_x)
    u2 = tf.reduce_sum(u ** 2, axis=1, keepdims=True)  # (batch,1)

    V_pred = V_model([x, t])
    V_t = V_target([x, t])
    V_jump = V_target([x + e, t]) - V_t

    residual = tf.squeeze(V_pred - (V_t + d2f_dh2 + physics.beta * u2 * V_jump + physics.theta * u2), axis=1)

    # Terminal condition
    V_pred_bc = V_model([tf.cast(xbcT, tf.float32), tf.cast(tbcT, tf.float32)])
    bc_res = tf.cast(VbcT, tf.float32) - V_pred_bc

    loss_bc = bc_weight * tf.reduce_mean(tf.square(bc_res))
    loss_pde = tf.reduce_mean(tf.square(residual))
    return loss_bc + loss_pde


def pinn_loss_control(
    V_model: tf.keras.Model,
    U_model: tf.keras.Model,
    x: tf.Tensor,
    t: tf.Tensor,
    e: tf.Tensor,
    physics: PhysicsParams,
) -> tf.Tensor:
    """Original pinn_loss2 (Hamiltonian minimization objective)."""
    x = tf.cast(x, tf.float32)
    t = tf.cast(t, tf.float32)
    e = tf.cast(e, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        V = V_model([x, t])
    V_x = tape.gradient(V, x)  # (batch, dim_x)

    u = U_model([x, t])
    u2 = tf.reduce_sum(u ** 2, axis=1, keepdims=True)  # (batch,1)

    V_jump = V_model([x + e, t]) - V  # (batch,1)

    # Per-sample Hamiltonian value (scalar per sample)
    H = tf.reduce_sum(V_x * u, axis=1, keepdims=True) + physics.theta * u2 + physics.beta * u2 * V_jump
    return tf.reduce_mean(H)


# -----------------------------
# Sampling
# -----------------------------

def _repeat_to_length(arr: np.ndarray, n: int) -> np.ndarray:
    """Repeat arr rows until length n."""
    if arr.shape[0] == n:
        return arr
    reps = int(np.ceil(n / arr.shape[0]))
    out = np.tile(arr, (reps, 1))
    return out[:n]


def sample_epoch_data(rng: np.random.Generator, physics: PhysicsParams, train: TrainParams, cov_j: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Generate one epoch of (x,t,e) and terminal boundary samples."""
    # interior
    t = rng.uniform(0.0, physics.T, size=(train.num_samples, 1)).astype(np.float32)
    x = rng.normal(loc=0.0, scale=np.sqrt(t), size=(train.num_samples, physics.dim_x)).astype(np.float32)
    mu = np.full((physics.dim_x,), physics.mu_j, dtype=np.float32)
    e = rng.multivariate_normal(mean=mu, cov=cov_j, size=train.num_samples).astype(np.float32)

    # terminal condition samples
    xbcT = rng.normal(0.0, physics.T, size=(train.nc_samples, physics.dim_x)).astype(np.float32)
    tbcT = (physics.T * np.ones((train.nc_samples, 1), dtype=np.float32))
    VbcT = (physics.lamb * np.sum(xbcT**2, axis=1, keepdims=True)).astype(np.float32)

    # match length train.num_samples for replay buffer tuples
    xbcT_m = _repeat_to_length(xbcT, train.num_samples)
    tbcT_m = _repeat_to_length(tbcT, train.num_samples)
    VbcT_m = _repeat_to_length(VbcT, train.num_samples)
    return x, t, e, xbcT_m, tbcT_m, VbcT_m


def make_test_set(rng: np.random.Generator, physics: PhysicsParams, train: TrainParams, cov_j: np.ndarray) -> Tuple[np.ndarray, ...]:
    t = rng.uniform(0.0, physics.T, size=(train.test_samples, 1)).astype(np.float32)
    x = rng.uniform(-1.5, 1.5, size=(train.test_samples, physics.dim_x)).astype(np.float32)
    mu = np.full((physics.dim_x,), physics.mu_j, dtype=np.float32)
    e = rng.multivariate_normal(mean=mu, cov=cov_j, size=train.test_samples).astype(np.float32)

    xbcT = rng.uniform(-1.5, 1.5, size=(train.test_samples, physics.dim_x)).astype(np.float32)
    tbcT = physics.T * np.ones((train.test_samples, 1), dtype=np.float32)
    VbcT = (physics.lamb * np.sum(xbcT**2, axis=1, keepdims=True)).astype(np.float32)
    return x, t, e, xbcT, tbcT, VbcT


# -----------------------------
# Training loop
# -----------------------------

class Trainer:
    def __init__(
        self,
        cfg: Config,
        strategy: tf.distribute.Strategy,
        sigma: np.ndarray,
        cov_j: np.ndarray,
    ):
        self.cfg = cfg
        self.strategy = strategy

        self.sigma_tf = tf.convert_to_tensor(sigma, dtype=tf.float32)
        self.cov_j = cov_j

        buffer_size = cfg.train.num_samples * cfg.train.buffer_mult
        self.replay = ReplayBuffer(buffer_size)

        self.rng = np.random.default_rng(cfg.run.seed)

        # test set + reference values
        xt, tt, et, xbc, tbc, vbc = make_test_set(self.rng, cfg.physics, cfg.train, cov_j)
        self.test = (xt, tt, et, xbc, tbc, vbc)

        self.lqr_ref = LQRReference(cfg.physics, cov_j)
        self.test_ref_values = np.asarray(
            [self.lqr_ref.value(float(tt[i, 0]), xt[i, :], sigma) for i in range(cfg.train.test_samples)],
            dtype=np.float32,
        ).reshape(-1, 1)

    def _make_dataset(self) -> tf.data.Dataset:
        x, t, e, xbcT, tbcT, VbcT = self.replay.as_arrays()
        ds = tf.data.Dataset.from_tensor_slices(
            (
                tf.convert_to_tensor(x, tf.float32),
                tf.convert_to_tensor(t, tf.float32),
                tf.convert_to_tensor(e, tf.float32),
                tf.convert_to_tensor(xbcT, tf.float32),
                tf.convert_to_tensor(tbcT, tf.float32),
                tf.convert_to_tensor(VbcT, tf.float32),
            )
        )
        ds = ds.shuffle(buffer_size=min(20000, self.replay.size()), seed=self.cfg.run.seed, reshuffle_each_iteration=True)
        ds = ds.batch(self.cfg.train.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return ds

    def train(self, V: tf.keras.Model, U: tf.keras.Model, Vt: tf.keras.Model) -> Dict[str, Any]:
        cfg = self.cfg
        p = cfg.physics
        tr = cfg.train

        metrics: Dict[str, list] = {
            "phase": [],
            "epoch": [],
            "train_loss": [],
            "test_loss": [],
            "test_mae_value": [],
            "wall_time_s": [],
        }

        best_loss = float("inf")
        best_weights = None

        xt, tt, et, xbc, tbc, vbc = self.test
        xt_tf = tf.convert_to_tensor(xt, tf.float32)
        tt_tf = tf.convert_to_tensor(tt, tf.float32)
        et_tf = tf.convert_to_tensor(et, tf.float32)
        xbc_tf = tf.convert_to_tensor(xbc, tf.float32)
        tbc_tf = tf.convert_to_tensor(tbc, tf.float32)
        vbc_tf = tf.convert_to_tensor(vbc, tf.float32)

        start_time = time.time()

        for phase in cfg.phases:
            with self.strategy.scope():
                if phase.optimizer == "adam":
                    optV = tf.keras.optimizers.Adam(learning_rate=phase.lr_value)
                    optU = tf.keras.optimizers.Adam(learning_rate=phase.lr_control)
                elif phase.optimizer == "adagrad":
                    optV = tf.keras.optimizers.Adagrad(learning_rate=phase.lr_value)
                    optU = tf.keras.optimizers.Adagrad(learning_rate=phase.lr_control)
                else:
                    raise ValueError(f"Unknown optimizer: {phase.optimizer}")

            @tf.function
            def _train_step(batch):
                x, t, e, xbcT, tbcT, VbcT = batch

                # --- value step
                with tf.GradientTape() as tapeV:
                    lossV = pinn_loss_value(
                        V, U, Vt, x, t, e, xbcT, tbcT, VbcT, p, self.sigma_tf, tr.bc_weight
                    )
                gradsV = tapeV.gradient(lossV, V.trainable_variables)
                gradsV, _ = tf.clip_by_global_norm(gradsV, phase.grad_clip_norm)
                optV.apply_gradients(zip(gradsV, V.trainable_variables))

                # --- control step
                with tf.GradientTape() as tapeU:
                    lossU = pinn_loss_control(V, U, x, t, e, p)
                gradsU = tapeU.gradient(lossU, U.trainable_variables)
                gradsU, _ = tf.clip_by_global_norm(gradsU, phase.grad_clip_norm)
                optU.apply_gradients(zip(gradsU, U.trainable_variables))

                return lossV + lossU

            def _distributed_train_step(batch):
                per_replica = self.strategy.run(_train_step, args=(batch,))
                return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

            for epoch in range(phase.epochs):
                # add fresh samples to replay buffer
                x, t, e, xbcT, tbcT, VbcT = sample_epoch_data(self.rng, p, tr, self.cov_j)
                self.replay.add_batch(x, t, e, xbcT, tbcT, VbcT)

                # update value target
                soft_update(Vt, V, tau=phase.target_tau)

                # train on replay buffer
                ds = self._make_dataset()
                dist_ds = self.strategy.experimental_distribute_dataset(ds)

                train_loss_sum = 0.0
                steps = 0
                for batch in dist_ds:
                    train_loss_sum += float(_distributed_train_step(batch))
                    steps += 1

                train_loss = train_loss_sum / max(steps, 1)

                # periodic evaluation
                if (epoch + 1) % phase.log_every == 0 or (epoch + 1) == phase.epochs:
                    test_loss = float(
                        pinn_loss_value(V, U, Vt, xt_tf, tt_tf, et_tf, xbc_tf, tbc_tf, vbc_tf, p, self.sigma_tf, tr.bc_weight)
                        + pinn_loss_control(V, U, xt_tf, tt_tf, et_tf, p)
                    )

                    # Track best
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_weights = (V.get_weights(), U.get_weights())

                    # MAE against reference value function
                    V_pred = V([xt_tf, tt_tf]).numpy()
                    mae = float(np.mean(np.abs(V_pred - self.test_ref_values)))

                    elapsed = time.time() - start_time
                    metrics["phase"].append(phase.name)
                    metrics["epoch"].append(epoch + 1)
                    metrics["train_loss"].append(train_loss)
                    metrics["test_loss"].append(test_loss)
                    metrics["test_mae_value"].append(mae)
                    metrics["wall_time_s"].append(elapsed)

                    print(
                        f"[{phase.name}] Epoch {epoch+1}/{phase.epochs} | "
                        f"train={train_loss:.6g} test={test_loss:.6g} mae(V)={mae:.6g} "
                        f"buffer={self.replay.size()} time={elapsed:.1f}s"
                    )

        # restore best weights (optional but matches original intent)
        if best_weights is not None:
            V.set_weights(best_weights[0])
            U.set_weights(best_weights[1])

        return {"metrics": metrics, "best_test_loss": best_loss}


# -----------------------------
# Post-training diagnostics
# -----------------------------

def evaluate_and_plot(cfg: Config, V: tf.keras.Model, U: tf.keras.Model, sigma: np.ndarray, cov_j: np.ndarray) -> None:
    p = cfg.physics
    ref = LQRReference(p, cov_j)

    tt = 0.0
    x1_min, x1_max = -2.5, 2.5
    n = 1000

    x_test = np.zeros((n, p.dim_x), dtype=np.float32)
    x_test[:, 0] = np.linspace(x1_min, x1_max, n).astype(np.float32)
    t_test = np.full((n, 1), tt, dtype=np.float32)

    V_pred = V([x_test, t_test]).numpy()
    U_pred = U([x_test, t_test]).numpy()

    V_ref = np.asarray([ref.value(tt, x_test[i], sigma) for i in range(n)], dtype=np.float32).reshape(-1, 1)
    U_ref = np.asarray([ref.control(tt, x_test[i]) for i in range(n)], dtype=np.float32)

    print("MAE_V =", float(np.mean(np.abs(V_pred - V_ref))))
    print("MAE_u =", float(np.mean(np.sqrt(np.sum((U_pred - U_ref) ** 2, axis=1)))))

    if not cfg.run.make_plots:
        return

    plt.figure()
    plt.plot(x_test[:, 0], V_pred)
    plt.plot(x_test[:, 0], V_ref, "--")
    plt.xlabel("$x_1$")
    plt.ylabel("Value function $\\mathcal{V}$")
    plt.savefig(f"{cfg.run.plot_prefix}1c.png")
    plt.close()

    plt.figure()
    plt.plot(x_test[:, 0], U_pred[:, 0])
    plt.plot(x_test[:, 0], U_ref[:, 0])
    plt.xlabel("$x_1$")
    plt.ylabel("Control $u_1$")
    plt.savefig(f"{cfg.run.plot_prefix}2c.png")
    plt.close()


# -----------------------------
# CLI / main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GPI-CBU training (cleaned).")
    ap.add_argument("--config", type=str, default=None, help="Path to TOML/JSON config file.")
    ap.add_argument("--seed", type=int, default=None, help="Override seed.")
    ap.add_argument("--no-plots", action="store_true", help="Disable plots.")
    ap.add_argument("--no-save", action="store_true", help="Disable saving weights.")
    ap.add_argument("--no-mirror", action="store_true", help="Disable MirroredStrategy.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_file(args.config) if args.config else Config()

    # Small CLI overrides (keep it simple)
    run = cfg.run
    if args.seed is not None:
        run = RunParams(**{**asdict(run), "seed": args.seed})
    if args.no_plots:
        run = RunParams(**{**asdict(run), "make_plots": False})
    if args.no_save:
        run = RunParams(**{**asdict(run), "save_weights": False})
    if args.no_mirror:
        run = RunParams(**{**asdict(run), "use_mirrored_strategy": False})
    cfg = Config(physics=cfg.physics, model=cfg.model, train=cfg.train, phases=cfg.phases, run=run)

    set_global_seeds(cfg.run.seed)

    if cfg.run.use_mirrored_strategy:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    if cfg.run.device_log:
        print("Strategy:", type(strategy).__name__)
        print("Replicas:", strategy.num_replicas_in_sync)

    # Sample sigma, cov_j (same logic as original, but centralized)
    rng = np.random.default_rng(cfg.run.seed)
    sigma = rng.normal(loc=0.0, scale=cfg.physics.sigma_scale, size=(cfg.physics.dim_x, cfg.physics.dim_w)).astype(np.float32)
    cov_j = np.diag(rng.uniform(low=0.0, high=cfg.physics.cov_j_max, size=cfg.physics.dim_x)).astype(np.float32)

    with strategy.scope():
        V, U, Vt = build_models(cfg)
        # build once
        dummy_x = tf.zeros((2, cfg.physics.dim_x), tf.float32)
        dummy_t = tf.zeros((2, 1), tf.float32)
        _ = V([dummy_x, dummy_t])
        _ = U([dummy_x, dummy_t])
        _ = Vt([dummy_x, dummy_t])

    trainer = Trainer(cfg, strategy, sigma=sigma, cov_j=cov_j)
    out = trainer.train(V, U, Vt)

    if cfg.run.save_weights:
        V.save_weights(f"{cfg.run.weights_prefix}1.weights.h5")
        U.save_weights(f"{cfg.run.weights_prefix}2.weights.h5")

    evaluate_and_plot(cfg, V, U, sigma=sigma, cov_j=cov_j)

    # Save metrics for later inspection
    with open("train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()


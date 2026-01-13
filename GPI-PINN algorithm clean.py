# -*- coding: utf-8 -*-
"""

GPI-PINN Algorithm applied on the Example (5.1) from Section 5 and derived from  Han et al. (2017)
Version of the GPI-PINN Algorithm 3 without the early terminal conditions of equations (3.9) and (3.10).
We consider a 10-dimensional problem with 9 dimensions for the state space and 1 dimension in time.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit

    

##### MC simulation of the value function and optimal control, see equations (5.3) and (5.4) #####

#function phi giving the terminal condition at each x
def phicontrol(x):
    out = np.log((np.sum(x**2, axis=1)+1)/2)
    return out

# MC value function (5.3)
def value1(T,t,x,lambd):
    WW = np.random.normal(0,np.sqrt(T-t), (m,len(x)))
    WW1 = np.sqrt(2)*WW
    for i in range(WW.shape[1]):
        WW1[:,i] = WW1[:,i] + x[i]
    WW11 = np.exp(-lambd*phicontrol(WW1))
    res1 = np.mean(WW11)
    return(-np.log(res1)/lambd)

# MC optimal control (5.4)
def control1(T,t,x,lambd, index):
    WW = np.random.normal(0,np.sqrt(T-t), (m,len(x)))
    WW1 = np.sqrt(2)*WW
    for i in range(WW.shape[1]):
        WW1[:,i] = WW1[:,i] + x[i]
    WW11 = np.exp(-lambd*phicontrol(WW1))
    res1 = np.mean(WW11)
    WW2 = WW11*(x[index]+np.sqrt(2)*WW[:,index])
    WW2 = WW2/(1+np.sum(WW1**2, axis=1))
    res2 = np.mean(WW2)
    out = (2*res2/(res1))
    return(-np.sqrt(lambd)*out)

# Computation of the MC solution at some point to derive Tables 5.1 and 5.2
xx=np.array([0.5,0,0,0,0,0,0,0,0])  #9 dimensions for the state space X
m=1000000 # number of MC simulations
true_pred = value1(1,0,xx,1)
true_control = control1(1,0,xx,1,0)
#true_control = -0.050495706762829384
#true_pred=2.0496778598180683


####### GPI-PINN Algorithm 3 based on the DGM NN (without early terminal conditons) ########

dimen = 9 #dimension of the state space X

# DGM NN for the optimal control u* (9-dimensional output)
class DGMCell_u(tf.keras.Model):
    def __init__(self, hidden_dim, n_layers=5, output_dim=dimen): 
        super(DGMCell_u, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n = n_layers

        self.sig_act = tf.keras.layers.Activation(tf.nn.tanh)

        self.Sw = tf.keras.layers.Dense(self.hidden_dim)
        self.Uz = tf.keras.layers.Dense(self.hidden_dim)
        self.Wsz = tf.keras.layers.Dense(self.hidden_dim)
        self.Ug = tf.keras.layers.Dense(self.hidden_dim)
        self.Wsg = tf.keras.layers.Dense(self.hidden_dim)
        self.Ur = tf.keras.layers.Dense(self.hidden_dim)
        self.Wsr = tf.keras.layers.Dense(self.hidden_dim)
        self.Uh = tf.keras.layers.Dense(self.hidden_dim)
        self.Wsh = tf.keras.layers.Dense(self.hidden_dim)
        self.Wf = tf.keras.layers.Dense(self.output_dim)

    def call(self, inputs):
        x, t = inputs
        xt = tf.concat([x,t], axis=1)
        S1 = self.sig_act(self.Sw(xt))
        out = S1
        for i in range(1, self.n):
            S = out
            Z = self.sig_act(self.Uz(xt) + self.Wsz(S))
            #G = self.sig_act(self.Ug(xt) + self.Wsg(S1))
            # typo in the code and paper
            G = self.sig_act(self.Ug(xt) + self.Wsg(S))
            R = self.sig_act(self.Ur(xt) + self.Wsr(S))
            H = self.Uh(xt) + self.Wsh(S * R)
            out = (1 - G) * H + Z * S
        out = self.Wf(out)
        return out

# DGM NN for the value function V (1-dimensional output)
class DGMCell_V(tf.keras.Model):
    def __init__(self, hidden_dim, n_layers=5, output_dim=1):
        super(DGMCell_V, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n = n_layers

        self.sig_act = tf.keras.layers.Activation(tf.nn.tanh)
        self.sig_out = tf.keras.layers.Activation(tf.nn.softplus)

        self.Sw = tf.keras.layers.Dense(self.hidden_dim)
        self.Uz = tf.keras.layers.Dense(self.hidden_dim)
        self.Wsz = tf.keras.layers.Dense(self.hidden_dim)
        self.Ug = tf.keras.layers.Dense(self.hidden_dim)
        self.Wsg = tf.keras.layers.Dense(self.hidden_dim)
        self.Ur = tf.keras.layers.Dense(self.hidden_dim)
        self.Wsr = tf.keras.layers.Dense(self.hidden_dim)
        self.Uh = tf.keras.layers.Dense(self.hidden_dim)
        self.Wsh = tf.keras.layers.Dense(self.hidden_dim)
        self.Wf = tf.keras.layers.Dense(self.output_dim)

    def call(self, inputs):
        x, t = inputs
        xt    = tf.concat([x,t], axis=1)
        S1 = self.sig_act(self.Sw(xt))
        out = S1
        for i in range(1, self.n):
            S = out
            Z = self.sig_act(self.Uz(xt) + self.Wsz(S))
            #G = self.sig_act(self.Ug(x) + self.Wsg(S1))
            # typo in the code and paper
            G = self.sig_act(self.Ug(xt) + self.Wsg(S))
            R = self.sig_act(self.Ur(xt) + self.Wsr(S))
            H = self.Uh(xt) + self.Wsh(S * R)
            out = (1 - G) * H + Z * S
        out = self.Wf(out)
        return out


#function to comupte by automatic differentiation the second order derivatives V_{x_i x_i}, i=1,...,d.
def compute_V_xx_terms(tape, V_x, x_tf):
    num_dimensions = V_x.shape[1]
    V_xx_terms = [tape.gradient(V_x[:, i], x_tf)[:, i] for i in range(num_dimensions)]
    return V_xx_terms

#First loss function defined in equation (3.2), such that the value NN satisfies the HJB equation
def pinn_loss1(model1, model2, x, t, xbcT, tbcT, VbcT, T, lambd):
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x_tf, t_tf])
        V_pred = model1([x_tf, t_tf]) #value NN
        dV = tape.gradient(V_pred, [x_tf, t_tf]) #first order derivatives
        V_x, V_t = dV[0], dV[1]
        V_xx_terms = compute_V_xx_terms(tape, V_x, x_tf) #second order derivatives

    u_pred = model2([x_tf, t_tf]) #control NN
    V_t = tf.reshape(V_t, shape=(x.shape[0],))
    
    #HJB residual in the interior domain with V_pred and u_pred
    residual = V_t + tf.reduce_sum(u_pred**2, axis=1) + 2*math.sqrt(lambd)*tf.reduce_sum(V_x*u_pred, axis=1) + sum(V_xx_terms)
    
    #terminal condition on the value function at T
    V_pred_bc = model1([xbcT, tbcT])
    residual_bc = tf.convert_to_tensor(VbcT, dtype=tf.float32) - V_pred_bc

    #200 and 20 are a proportionality factor to fine-tune, see remarks below eq. (3.6) and (3.7)
    loss1 = 200 * tf.reduce_sum(tf.math.square(residual_bc)) 
    loss2 = 20*tf.reduce_sum(tf.math.square(residual))
    loss = loss1 + loss2

    return loss


#Second loss function for the minimization of the Hamiltonian w.r.t the control, defined in equation (3.4)  
def pinn_loss2(model1, model2,x,t, T,lambd):
    x_tf=tf.convert_to_tensor(x, dtype=tf.float32)# Create a Tensorflow variable
    t_tf = tf.convert_to_tensor(t, dtype=tf.float32)

    with tf.GradientTape() as tape1 :
              tape1.watch([x_tf])
              V_pred = model1([x_tf,t_tf]) #value NN
              dV =  tape1.gradient(V_pred, [x_tf])
              V_x = dV[0]

    u_pred = model2([x_tf,t_tf]) #control NN
    
    #minimization of the Hamiltonian with respect to the control u_pred (equation 3.4) :
    residual = tf.reduce_sum(u_pred**2, axis=1) + 2*math.sqrt(lambd)*tf.reduce_sum(V_x*u_pred, axis=1)
    loss = tf.reduce_sum(residual) 

    return loss



#Training procedure via batch gradient descent, as described in Section 4, p.15
def train(model1,model2,T,lambd, epochs, num_samples, nc_samples, lr=0.001, batch_size=128):
    optimizer1   = tf.keras.optimizers.Adam(learning_rate=lr)
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr)
    best_loss = float('inf')
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
        #at each epoch, I generate a new dataset -> gives slightly better results than with always the same dataset
        t=np.random.uniform(0,T,num_samples).reshape(-1,1)
        x = [*map(lambda x: np.random.normal(0,np.sqrt(x),dimen), t)]  #see p.17 for the probability laws used to generate the data
        xx = x[0].reshape(1,-1)
        for i in range(len(x)) :
          xx = np.concatenate((xx, x[i].reshape(1,-1)), axis=0)
        x=xx[1:(num_samples+1),]
        
        #Generate terminal data at  T :
        xbcTemp = [np.random.normal(0, np.sqrt(T), nc_samples).reshape(-1, 1) for _ in range(dimen)]
        xbcT = np.concatenate(xbcTemp, axis=1)
        tbcT    = T*np.ones(shape=(nc_samples,1))
        VbcT = phicontrol(xbcT).reshape(-1,1)#terminal condition for the value function V
        
        # Shuffle the data indices for each epoch
        indices = np.random.permutation(num_samples)
        total_loss = 0.0
        for batch in range(num_batches):
            # Retrieve the indices for the current batch
            batch_indices = indices[batch * batch_size: (batch + 1) * batch_size]
            # Get the batch data
            x_batch = x[batch_indices,:]
            t_batch = t[batch_indices]
            xbcT_batch = xbcT[batch_indices]
            tbcT_batch = tbcT[batch_indices]
            VbcT_batch = VbcT[batch_indices]
            #Phase 2, Step 1 of Algorithm 3 (training of the Value NN)
            with tf.GradientTape() as tape:
                loss1 = pinn_loss1(model1, model2,x_batch,t_batch,xbcT_batch, tbcT_batch, VbcT_batch, T=T, lambd = lambd)
            gradients1 = tape.gradient(loss1, model1.trainable_variables)
            optimizer1.apply_gradients(zip(gradients1, model1.trainable_variables))
            total_loss += loss1

        indices = np.random.permutation(num_samples) #also reshuffle the data before Step 2
        for batch in range(num_batches):
            # Retrieve the indices for the current batch
            batch_indices = indices[batch * batch_size: (batch + 1) * batch_size]
            # Get the batch data
            x_batch = x[batch_indices]
            t_batch = t[batch_indices]
            #Phase 2, Step 2 of the Algorithm 3 (training of the optimal control NN)
            with tf.GradientTape() as tape2:
                loss2 =  pinn_loss2(model1, model2,x_batch,t_batch, T=T, lambd = lambd)
            gradients2 = tape2.gradient(loss2, model2.trainable_variables)
            optimizer2.apply_gradients(zip(gradients2, model2.trainable_variables))
            total_loss += loss2
        avg_loss = total_loss / (2*num_batches)
        if avg_loss < best_loss:
            best_weights1 = model1.get_weights()
            best_weights2 = model2.get_weights()
            best_loss = avg_loss
            
        if (epoch+1) % 10 == 0:
            print("Epoch {}/{}: Loss = {}".format(epoch+1, epochs, avg_loss.numpy()))
    return best_weights1, best_weights2, best_loss

#The function train2 is similar as train1 but uses Adagrad instead of Adam method for updating the weights.
def train2(model1,model2,T,lambd, epochs, num_samples, nc_samples, lr=0.001, batch_size=128):
    optimizer1   = tf.keras.optimizers.experimental.Adagrad(learning_rate=lr)
    optimizer2 = tf.keras.optimizers.experimental.Adagrad(learning_rate=lr)
    best_loss = float('inf')
    num_batches = num_samples // batch_size
     
    for epoch in range(epochs):
        t=np.random.uniform(0,T,num_samples).reshape(-1,1)
        x = [*map(lambda x: np.random.normal(0,np.sqrt(x),dimen), t)]
        xx = x[0].reshape(1,-1)
        for i in range(len(x)) :
          xx = np.concatenate((xx, x[i].reshape(1,-1)), axis=0)
        x=xx[1:(num_samples+1),]
        
        #Generate terminal data at terminal time T :
        xbcTemp = [np.random.normal(0, np.sqrt(T), nc_samples).reshape(-1, 1) for _ in range(dimen)]
        xbcT = np.concatenate(xbcTemp, axis=1)
        tbcT    = T*np.ones(shape=(nc_samples,1))
        VbcT = phicontrol(xbcT).reshape(-1,1)#terminal condition for the value function 
        
        # Shuffle the data indices for each epoch
        indices = np.random.permutation(num_samples)
        total_loss = 0.0
        for batch in range(num_batches):
            # Retrieve the indices for the current batch
            batch_indices = indices[batch * batch_size: (batch + 1) * batch_size]
            # Get the batch data
            x_batch = x[batch_indices]
            t_batch = t[batch_indices]
            xbcT_batch = xbcT[batch_indices]
            tbcT_batch = tbcT[batch_indices]
            VbcT_batch = VbcT[batch_indices]
            with tf.GradientTape() as tape:
                loss1 = pinn_loss1(model1, model2,x_batch,t_batch,xbcT_batch, tbcT_batch, VbcT_batch, T=T, lambd = lambd)
            gradients1 = tape.gradient(loss1, model1.trainable_variables)
            optimizer1.apply_gradients(zip(gradients1, model1.trainable_variables))
            total_loss += loss1
    
        indices = np.random.permutation(num_samples)
        for batch in range(num_batches):
            # Retrieve the indices for the current batch
            batch_indices = indices[batch * batch_size: (batch + 1) * batch_size]
            # Get the batch data
            x_batch = x[batch_indices]
            t_batch = t[batch_indices]
            with tf.GradientTape() as tape2:
                loss2 =  pinn_loss2(model1, model2,x_batch,t_batch, T=T, lambd = lambd)
            gradients2 = tape2.gradient(loss2, model2.trainable_variables)
            optimizer2.apply_gradients(zip(gradients2, model2.trainable_variables))
            total_loss += loss2
        avg_loss = total_loss / (2*num_batches)
        if avg_loss < best_loss:
            best_weights1 = model1.get_weights()
            best_weights2 = model2.get_weights()
            best_loss = avg_loss

        if (epoch+1) % 10 == 0:
            print("Epoch {}/{}: Loss = {}".format(epoch+1, epochs, avg_loss.numpy()))
    return best_weights1, best_weights2, best_loss
# tester à la place Howard algorithm : eq.(1.21) dans pin_loss2 !!!


###### Application #####
lr    = 0.005  #learning rate
epochs= 200
num_samples = 5000  #training size (interior of the domain)
nc_samples = 5000   #training size (at terminal condition)
T = 1 #terminal time
lambd = 1  #parameter, see Section 5.

model1 = DGMCell_V(50, 3,1)  #value NN with 3 layers and 50 hidden neurons
model2 = DGMCell_u(50,3,dimen)  #optimal control NN

tic = timeit.default_timer()
res_fin = train(model1, model2, T, lambd, epochs, num_samples, nc_samples,lr)
toc = timeit.default_timer() 
h1 = toc-tic

tic = timeit.default_timer()
res_fin = train2(model1, model2, T, lambd, epochs, num_samples, nc_samples,lr)
toc = timeit.default_timer() 
h2 = toc-tic
   
#x_1 in [-1.5,1.5], x_j = 0 for j=2,...,9 and t=0 (see Figure 5.1-5.2)
x1_min=-1.5
x1_max = 1.5
dim_test = 501 
x_test = np.zeros((dim_test, dimen))
x_test[:, 0] = np.linspace(x1_min, x1_max, dim_test)
x_test = np.concatenate([x_test[:, i].reshape(-1, 1) for i in range(dimen)], axis=1)
t_1  = (0)*np.ones(dim_test).reshape(-1, 1)

#GPI-PINN approximation
V_1_pred  = model1([x_test,t_1])
V_1_rate = model2([x_test,t_1])

#MC approximation
tt=0 #time 0
m = 500000 #number of MC samples
res_valuex= [*map(lambda x: value1(T,tt,x,lambd), x_test)]
res_controlx= [*map(lambda x: control1(T,tt,x,lambd,0), x_test)]
#analosss = np.mean((model1([x_test,t_1])-res_valuex)**2)
#analosss

#Plots
plt.figure()
plt.plot(x_test[:,0], V_1_pred)
plt.plot(x_test[:,0], res_valuex)
plt.xlabel("$x_1$")
plt.ylabel("Value function $\mathcal{V}$")
plt.figure()

plt.plot(x_test[:,0], V_1_rate)
plt.plot(x_test[:,0], res_controlx)
plt.xlabel("$x$")
plt.ylabel("Optimal Control $u*$")
plt.show()


   

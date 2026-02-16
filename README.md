# Deep-Learning-for-CT-Stochastic-Control-with-Jumps
We introduce a model-based deep-learning approach to solve finite-horizon continuous-time stochastic control problems with jumps. We iteratively train two neural networks: one to represent the optimal policy and the other to approximate the value function. Leveraging a continuous-time version of the dynamic programming principle, we derive two different training objectives based on the Hamilton-Jacobi-Bellman equation, ensuring that the networks capture the underlying stochastic dynamics. Empirical evaluations on different problems illustrate the accuracy and scalability of our approach, demonstrating its effectiveness in solving complex high-dimensional stochastic control tasks.

How to run:

python GPI-CBU_clean.py

or with a config file:

python GPI-CBU_clean.py --config gpi_cbu_config.example.toml

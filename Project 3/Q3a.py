import numpy as np 
import scipy as sp
from scipy.stats import multivariate_normal, gamma
import matplotlib.pyplot as plt
import Sub_data as Sub
from scipy.integrate import solve_ivp, RK45

var_beta = 8.0/3
var_rho = 28.0
var_sigma = 10.0
sigma_sq = 4
n_dim = 3
n_ensem = 100          # M

t_start = 0
delta_t = 0.05
t_end = 10
num_t = int((t_end - t_start) / delta_t)
Z_0 = np.array([0, 1, 0]) 
# RK_func = lambda z_vec: Sub.int_func(z_vec)
t_vec = np.arange(t_start, t_end, delta_t)
RK_output = solve_ivp(Sub.int_func, (t_start, t_end), Z_0, method = 'RK45', t_eval = t_vec, first_step = delta_t, max_step = delta_t)
Z_vec = RK_output.y
t_eff = RK_output.t

Y_mat = np.zeros(np.shape(Z_vec), dtype = float)
for i1 in range(Z_vec.shape[1]):
    Y_mat[:, i1] = multivariate_normal.rvs(mean = Z_vec[:, i1], cov = sigma_sq * np.eye(n_dim))         # Observations

## Ensemble Kalman Filter

t_state = 0
z_state = np.zeros((n_dim, n_ensem, num_t), dtype = float)

for j1 in range(num_t):
    for k1 in range(n_ensem):
        RK_output = solve_ivp(Sub.int_func, (t_state, t_state + delta_t), z_state , method = 'RK45')
        z_state_unhat = RK_output.y[:, -1]
        cov_rec = np.cov(z_state_unhat)
    print(1)



import numpy as np 
import scipy as sp
from scipy.stats import multivariate_normal, gamma
import matplotlib.pyplot as plt
import Sub_data as Sub
from scipy.integrate import solve_ivp, RK45
from numpy.linalg import inv, norm

var_beta = 8/3
var_rho = 28.0
var_sigma = 10.0
sigma_sq = 4
n_dim = 3
# n_ensem = np.arange(100, 2000, 200)          # M
n_ensem = np.arange(25, 1000, 100)          # M

num_en = len(n_ensem)

t_start = 0
delta_t = 0.05
delta_t_obs = 2 * delta_t
t_end = 10
num_t = int((t_end - t_start) / delta_t) 

# Z_0 = np.array([1, 1, 1]) 
Z_0 = multivariate_normal.rvs(mean = np.zeros(n_dim), cov = np.eye(n_dim))  
t_vec = np.arange(t_start, t_end + delta_t/2, delta_t)
RK_output = solve_ivp(Sub.int_func, (t_start, t_end), Z_0, method = 'RK45', t_eval = t_vec, first_step = delta_t, max_step = delta_t)
Z_vec = RK_output.y
t_eff = RK_output.t * 2

gamma_mat = sigma_sq * np.eye(n_dim)
Y_mat = np.zeros(np.shape(Z_vec), dtype = float)

for i1 in range(Z_vec.shape[1]):
    Y_mat[:, i1] = multivariate_normal.rvs(mean = Z_vec[:, i1], cov = gamma_mat)         # Observations

## Ensemble Kalman Filter
average_RMSE_vec = np.zeros(num_en)
for i1 in range(num_en):

    z_state = np.zeros((n_dim, n_ensem[i1], num_t), dtype = float)
    z_hat_state = np.zeros((n_dim, n_ensem[i1], num_t), dtype = float)
    RMSE_vec = np.zeros(num_t, dtype = float)
    mean_z_vec = np.zeros((n_dim, num_t), dtype = float)

    # Initial conditions
    for i2 in range(n_ensem[i1]):
        # z_state[:, i2, 0] = multivariate_normal.rvs(mean = np.zeros(n_dim), cov = np.eye(n_dim))         # Observations
        z_state[:, i2, 0] = multivariate_normal.rvs(mean = Z_0, cov = np.eye(n_dim))         # Observations

    # Forecast
    for j1 in range(num_t - 1):
        for k1 in range(n_ensem[i1]):
            RK_output = solve_ivp(Sub.int_func, (t_eff[j1], t_eff[j1 + 1]), z_state[:, k1, j1] , method = 'RK45')
            z_hat_state[:, k1, j1 + 1] = RK_output.y[:, -1]

        cov_mat_np1 = np.cov(z_hat_state[:, :, j1 + 1])

    # Analysis
        gain_mat = np.dot(cov_mat_np1, inv(gamma_mat + cov_mat_np1)) 

        for k2 in range(n_ensem[i1]):
            noise_vec = multivariate_normal.rvs(mean = np.zeros(n_dim), cov = gamma_mat)
            z_state[:, k2, j1 + 1] = z_hat_state[:, k2, j1 + 1] + np.inner(gain_mat, Y_mat[:, j1 + 1] - z_hat_state[:, k2, j1 + 1] + noise_vec) 

        mean_z_vec[:, j1] = np.mean(z_state[:, :, j1 + 1], axis = 1)
        RMSE_vec[j1] = norm(mean_z_vec[:, j1] - Z_vec[:, j1 + 1]) / np.sqrt(3)
    
    average_RMSE_vec[i1] = np.mean(RMSE_vec[int(num_t/2):])
    print(n_ensem[i1])

fig, ax = plt.subplots(figsize = (5,5))
ax.plot(n_ensem, average_RMSE_vec, color = 'maroon', label = 'RMSE')
ax.plot([n_ensem[0], n_ensem[-1]], [2, 2], color = 'navy', label = 'Standard Deviation')
ax.set_xlabel('Size of ensemble, M')
ax.set_ylabel('RMSE')
ax.set_title('RMSE with $\delta$t = 0.05, with ' + str(num_t) +' time steps')
ax.set_xbound(n_ensem[0], n_ensem[-1] )
ax.set_ybound(0, int(average_RMSE_vec[0] + 1))
ax.legend()
plt.show()
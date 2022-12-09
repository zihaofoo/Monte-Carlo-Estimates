import numpy as np 
import scipy as sp
from scipy.stats import multivariate_normal, multinomial
import matplotlib.pyplot as plt
import Sub_data as Sub
from scipy.integrate import solve_ivp
from numpy.linalg import norm

var_beta = 8/3
var_rho = 28.0
var_sigma = 10.0
sigma_sq = 4
n_dim = 3
n_ensem = [100]          # M
# n_ensem = np.arange(25, 300, 50)          # M
sigma_noise_sq = 1E-4

num_en = np.size(n_ensem)

t_start = 0
delta_t = 0.05
t_end = 10
num_t = int((t_end - t_start) / delta_t) 
Z_0 = np.array([0, 1, 0]) 
#Z_0 = multivariate_normal.rvs(mean = np.zeros(n_dim), cov = np.eye(n_dim))  
t_vec = np.arange(t_start, t_end + delta_t/2, delta_t)

RK_output = solve_ivp(Sub.int_func, (t_start, t_end), Z_0, method = 'RK45', t_eval = t_vec, first_step = delta_t, max_step = delta_t)
Z_vec = RK_output.y
t_eff = RK_output.t * 2

gamma_mat = sigma_sq * np.eye(n_dim)
Y_mat = np.zeros(np.shape(Z_vec), dtype = float)

for i1 in range(Z_vec.shape[1]):
    Y_mat[:, i1] = multivariate_normal.rvs(mean = Z_vec[:, i1], cov = gamma_mat)         # Observations

average_RMSE_vec = np.zeros(num_en)

for i1 in range(num_en):
    weight_vec = np.ones(n_ensem[i1]) / n_ensem[i1]

    z_state = np.zeros((n_dim, n_ensem[i1], num_t), dtype = float)
    z_hat_state = np.zeros((n_dim, n_ensem[i1], num_t), dtype = float)
    RMSE_vec = np.zeros(num_t, dtype = float)
    mean_z_vec = np.zeros((n_dim, num_t), dtype = float)

   # Initial conditions
    for i2 in range(n_ensem[i1]):
        z_state[:, i2, 0] = multivariate_normal.rvs(mean = np.zeros(n_dim), cov = np.eye(n_dim))         # Observations

    # Particle Filterings
    for j1 in range(num_t - 1):     # Time stepping

        for k1 in range(n_ensem[i1]):
            RK_output = solve_ivp(Sub.int_func, (t_eff[j1], t_eff[j1 + 1]), z_state[:, k1, j1] , method = 'RK45')
            z_hat_state[:, k1, j1 + 1] = RK_output.y[:, -1]
            z_state[:, k1, j1 + 1] = multivariate_normal.rvs(mean = z_hat_state[:, k1, j1 + 1], cov = sigma_noise_sq * np.eye(n_dim)) 
            weight_vec[k1] = weight_vec[k1] * multivariate_normal.pdf(Y_mat[:, j1], mean = z_state[:, k1, j1 + 1], cov = sigma_sq * np.eye(n_dim))
        
        weight_norm_vec = weight_vec / np.sum(weight_vec)
        

        ## Re-sampling sequence
        ESS = 1 / np.sum(weight_norm_vec**2)

        if ESS >= n_ensem[i1] / 10:
            index_vec = multinomial.rvs(n_ensem[i1], weight_norm_vec)
            x_vec = np.zeros((n_dim, n_ensem[i1]))
            w_vec = np.zeros(n_ensem[i1])
            index_state = 0
            for l1 in range(len(index_vec)):
                if index_vec[l1] == 0:
                    continue
                else:
                    repeat_var = index_vec[l1]
                    for l2 in range(repeat_var):
                        x_vec[:, index_state] = z_state[:, l1, j1 + 1]
                        w_vec[index_state] = weight_norm_vec[l1]
                        index_state = index_state + 1 
            z_state[:, :, j1 + 1] = x_vec
            weight_vec = w_vec
        
        # Calculating RMSE and covariance
        weight_vec = weight_vec / np.sum(weight_vec)
        
        # print(weight_vec)
        mean_z_vec[:, j1] = np.mean(z_state[:, :, j1 + 1] * weight_vec, axis = 1)
        RMSE_vec[j1] = norm(mean_z_vec[:, j1] - Z_vec[:, j1 + 1]) / np.sqrt(3)
    average_RMSE_vec[i1] = np.mean(RMSE_vec)
    
    # print(n_ensem[i1])
    print(mean_z_vec)
        
fig, ax = plt.subplots(figsize = (5,5))
ax.plot(n_ensem, average_RMSE_vec, color = 'maroon', label = 'RMSE')
ax.plot([n_ensem[0], n_ensem[-1]], [2, 2], color = 'navy', label = 'Standard Deviation')
ax.set_xlabel('Size of ensemble, M')
ax.set_ylabel('RMSE')
ax.set_title('RMSE with $\delta$t = 0.05, with ' + str(num_t) +' time steps')
# ax.set_xbound(n_ensem[0], n_ensem[-1])
# ax.set_ybound(0, int(average_RMSE_vec[0] + 1))
ax.legend()
plt.savefig('Q3b-1.pdf')

ax1 = plt.axes(projection='3d')
ax1.plot3D(Z_vec [0, :], Z_vec [1, :], Z_vec [2, :], 'green')
ax1.plot3D(mean_z_vec[0, :], mean_z_vec[1, :], mean_z_vec[2, :], 'red')

plt.show()
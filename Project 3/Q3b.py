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
n_ensem = np.arange(25, 1000, 50)          # M

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
plt.savefig('Q3a.pdf')
plt.show()
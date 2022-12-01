import pdb
import numpy as np 
import scipy as sp
import Sub_data as Sub
import pandas as pd
from scipy.stats import multivariate_normal, gamma
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

## Variable Definition
ktrue, xgrid, xobserved, Uobserved = Sub.load_data()

num_deg = 10                    # Value of d
num_MCMC = 1500                 # Number of MC steps
sigma_ep = np.sqrt(10**(-4))
F_vec = -1.0
s_vec = Sub.source(x_vec = xobserved)
rightbc_vec = 1.0
cov_ep_mat = sigma_ep**2 * np.eye(num_deg, dtype = float)
burn_in = int(num_MCMC / 2)
gamma_a = 1E0
gamma_b = 1E-4
gamma_vec = np.array([gamma_a, gamma_b])

## Adaptive Monte Carlo
_, z_mat = Sub.adaptive_MC(num_MCMC, num_deg, xobserved, Uobserved, sigma_epsilon = sigma_ep, right_bc = rightbc_vec, epsil = 0.05)
_, z_mat_2, sigma_sq = Sub.adaptive_MC_epsilon(num_MCMC, num_deg, xobserved, Uobserved, right_bc = rightbc_vec, epsil = 0.05, gamma_1 = gamma_vec[0], gamma_2 = gamma_vec[1])

k_vec = np.exp(Sub.Y_n(xobserved, z_mat[-1, :], n = num_deg))
# k_org = np.exp(Sub.Y_n(xobserved, z_org[0, :], n = num_deg))

num_rows, num_cols = z_mat.shape
k_mat = np.zeros((num_rows, len(xobserved)))
k_mat_2 = np.zeros((num_rows, len(xobserved)))
u_mat = np.zeros((num_rows, len(xobserved)))
u_mat_2 = np.zeros((num_rows, len(xobserved)))

for k1 in range(num_rows):
    k_mat[k1, :] = np.exp(Sub.Y_n(xobserved, z_mat[k1, :], n = num_deg))
    k_mat_2[k1, :] = np.exp(Sub.Y_n(xobserved, z_mat_2[k1, :], n = num_deg))
    u_mat[k1, :] = Sub.diffusioneqn(xobserved, F = F_vec, k = k_mat[k1, :], source = s_vec, rightbc = rightbc_vec)
    u_mat_2[k1, :] = Sub.diffusioneqn(xobserved, F = F_vec, k = k_mat_2[k1, :], source = s_vec, rightbc = rightbc_vec)

u_vec = u_mat[:, 0]
u_vec_2 = u_mat_2[:, 0]

k_mean = np.mean(k_mat, axis = 0)
k_std = np.sqrt(np.var(k_mat, axis = 0))

# Data Visualization
fig_a, (ax_a, ax_b) = plt.subplots(nrows = 1, ncols = 2)
ax_a.hist(u_vec[burn_in:], bins = 20, color = 'maroon', label = 'Deterministic $\sigma^2$')
ax_b.hist(u_vec_2[burn_in:], bins = 20, color = 'navy', label = 'Variable $\sigma^2$')
ax_a.legend(frameon = False)
ax_b.legend(frameon = False)
ax_a.set_ylabel('Frequency')
ax_a.set_xlabel('u(x=0)') 
ax_b.set_xlabel('u(x=0)') 
ax_a.set_title('Deterministic sigma')
ax_b.set_title('Variable sigma')

plt.tight_layout()
plt.show()
# 


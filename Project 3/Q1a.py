import pdb
import numpy as np 
import scipy as sp
import Sub_data as Sub
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

## Variable Definition
ktrue, xgrid, xobserved, Uobserved = Sub.load_data()

num_deg = 100                # Value of d
num_MCMC = 100             # Number of MC steps
sigma_ep = np.sqrt(10**(-4))
F_vec = -1.0
s_vec = Sub.source(x_vec = xobserved)
rightbc_vec = 1.0
cov_ep_mat = sigma_ep**2 * np.eye(num_deg, dtype = float)

## Adaptive Monte Carlo
z_mat = Sub.adaptive_MC(num_MCMC, num_deg, xobserved, Uobserved, sigma_epsilon = sigma_ep, right_bc = rightbc_vec)
k_vec = np.exp(Sub.Y_n(xobserved, z_mat[-1, :], n = num_deg))

# Data Visualization
fig, ax = plt.subplots(figsize = (5,5))
ax.plot(xgrid, ktrue, label = 'True Value', color = 'maroon')
ax.plot(xobserved, k_vec, label = 'Adaptive Monte Carlo', color = 'navy')
ax.legend()
plt.show()
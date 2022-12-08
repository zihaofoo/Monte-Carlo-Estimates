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

num_deg = 25                    # Value of d
num_MCMC = 1000                 # Number of MC steps
sigma_ep = np.sqrt(10**(-4))
F_vec = -1.0
s_vec = Sub.source(x_vec = xobserved)
rightbc_vec = 1.0
cov_ep_mat = sigma_ep**2 * np.eye(num_deg, dtype = float)
burn_in = int(num_MCMC / 2)

## Adaptive Monte Carlo
mean_var = Sub.variance_min(num_MCMC, num_deg, xobserved, Uobserved, sigma_epsilon = sigma_ep, right_bc = rightbc_vec, epsil = 0.05)

fig, ax = plt.subplots(figsize = (5,5))
ax.plot(xobserved[:-1], mean_var, color = 'maroon', marker = 'v', markersize = 8)
ax.set_xlabel('X coordinate')
ax.set_ylabel('Spatially averaged posterior variance of k(x)')
plt.savefig('Q1d.pdf')
plt.show()
print(mean_var)

import pdb
import numpy as np 
import scipy as sp
import Sub_data as Sub
from scipy.stats import multivariate_normal

## Variable Definition
ktrue, xgrid, xobserved, Uobserved = Sub.load_data()

num_deg = 100                # Value of d
num_MCMC = 1000             # Number of MC steps
sigma_ep = np.sqrt(10**(-4))
F_vec = -1.0
s_vec = Sub.source(x_vec = xobserved)
rightbc_vec = 1.0
cov_ep_mat = sigma_ep**2 * np.eye(num_deg, dtype = float)

## Monte Carlo
cov_Z_init = np.eye(num_deg, dtype = float)                 # Covariance of Z (initial)
mu_Z_init = np.zeros((num_deg), dtype = float)              # Mean of Z (initial)
Z_init = multivariate_normal.rvs(mean = mu_Z_init, cov = cov_Z_init)         # Initial sample of Z
# Z_init = Z_init.reshape(len(Z_init), 1)

## Drawing from proposal
ep_proposal = 0.1
s_d = 2.4**2 / num_deg
cov_q_mat = s_d * np.cov(Z_init) + s_d * ep_proposal * np.eye(num_deg)

Z_proposal = multivariate_normal.rvs(mean = Z_init, cov = cov_Z_init)       # Drwaing from proposal distribution q
k_n = np.exp(Sub.Y_n(x = xobserved, Z = Z_init, n = num_deg, muY = 1))
u_vec = Sub.diffusioneqn(xgrid = xobserved, F = F_vec, k = k_n, source = s_vec, rightbc = rightbc_vec)

z_vec =  multivariate_normal.rvs(mean = mu_Z_init, cov = cov_Z_init) 
l_vec = multivariate_normal.rvs(mean = mu_Z_init, cov = cov_Z_init) 

z_vec = np.ones(150) 
posterior = Sub.PDF_posterior(z_vec, xobserved, Uobserved, sigma_ep, rightbc_vec)
print(posterior)
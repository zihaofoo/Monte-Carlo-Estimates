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

num_deg = 10               # Value of d
num_MCMC = 5000               # Number of MC steps
sigma_ep = np.sqrt(10**(-4))
F_vec = -1.0
s_vec = Sub.source(x_vec = xobserved)
rightbc_vec = 1.0
cov_ep_mat = sigma_ep**2 * np.eye(num_deg, dtype = float)

## Adaptive Monte Carlo
z_mat, z_org = Sub.adaptive_MC(num_MCMC, num_deg, xobserved, Uobserved, sigma_epsilon = sigma_ep, right_bc = rightbc_vec, epsil = 0.05)
k_vec = np.exp(Sub.Y_n(xobserved, z_mat[-1, :], n = num_deg))
k_org = np.exp(Sub.Y_n(xobserved, z_org[0, :], n = num_deg))

num_rows, num_cols = z_mat.shape
k_mat = np.zeros((num_rows, len(xobserved)))

cov_Z_init = np.eye(num_deg, dtype = float)                 # Covariance of Z (initial)
mu_Z_init = np.zeros((num_deg), dtype = float)              # Mean of Z (initial)
z_init_mat = np.zeros((501, num_deg), dtype = float)
k_init_mat = np.zeros((num_rows, len(xobserved)))

for j1 in range(501):
    z_init_mat[j1, :] = multivariate_normal.rvs(mean = mu_Z_init, cov = cov_Z_init)

for k1 in range(num_rows):
    k_mat[k1, :] = np.exp(Sub.Y_n(xobserved, z_mat[k1, :], n = num_deg))
    k_init_mat[k1, :] = np.exp(Sub.Y_n(xobserved, z_init_mat[k1, :], n = num_deg))

k_mean = np.mean(k_mat, axis = 0)
k_std = np.sqrt(np.var(k_mat, axis = 0))

## Data Processing
list_index = np.linspace(0, num_deg, num_deg, endpoint = True)
R_h = np.zeros((len(list_index), len(z_mat[:, 0])))
for j1 in range(len(list_index)):
    z_0 = z_mat[:, j1]
    z_bar = np.mean(z_0)
    C_0 = np.mean((z_0 - z_bar)**2)

    h_vec = np.arange(1, len(z_0) + 1, 1)
    C_h = np.zeros(len(z_0))

    for h in range(len(z_0)):
        sum_h = np.zeros(len(z_0) - h)
        for i1 in range(len(z_0) - h):
            sum_h[i1] = (z_0[i1] - z_bar) * (z_0[i1 + h] - z_bar)

        C_h[h] = np.mean(sum_h)

    R_h[j1,:] = C_h / C_0

# Covariance field calculation
x_grid_vec, y_grid_vec = np.meshgrid(xobserved, xobserved)
Cov_mat = np.cov(np.log(k_mat), rowvar = False)
Cov_init_mat = np.cov(np.log(k_init_mat), rowvar = False)

# Data Visualization
fig, ax = plt.subplots(figsize = (5,5))
ax.plot(xgrid, np.log(ktrue), label = 'True Value', color = 'maroon')
# ax.plot(xobserved, k_org, label = 'Initial Guess', color = 'olive', marker = '^', markersize = 8)
ax.plot(xobserved, np.log(k_mean), label = 'Adaptive Monte Carlo - Mean', color = 'navy', marker = '^', markersize = 8)
ax.plot(xobserved, np.log(k_mean + k_std), label = 'Adaptive Monte Carlo - +SD', color = 'olive', linestyle = '-.')
ax.plot(xobserved, np.log(k_mean - k_std), label = 'Adaptive Monte Carlo - -SD', color = 'peru', linestyle = '-.' )

ax.legend(frameon = False)
ax.set_ylabel('log Permeability, ln k(x)')
ax.set_xlabel('X coordinate, x') 
ax.set_xbound(lower = xobserved[0], upper = xobserved[-1])
plt.savefig('Q1a-1.pdf')

n_vec = np.arange(0, 501, 1)
fig1, ax1 = plt.subplots(figsize = (5,5))
for i1 in range(num_deg):
    if np.mod(i1, 4) == 0:
        ax1.plot(n_vec, z_mat[:, i1], label = 'Z_' + str(i1), linewidth = 1)

ax1.set_ylabel('Z value, Z')
ax1.set_xlabel('Number of AM steps, n') 
ax1.set_xbound(lower = n_vec[0], upper = n_vec[-1])
ax1.legend(ncol = 5, frameon = False, loc = 'upper right')
# ax1.set_ylabel('Permeability, k(x)')
# ax1.set_xlabel('X coordinate, x') 
plt.savefig('Q1a-2.pdf')


fig2, ax2 = plt.subplots(figsize = (5,5))
for j1 in range(len(R_h)):
    ax2.plot(h_vec, R_h[j1, :], label = 'Z_' + str(j1))
ax2.set_ylabel('Autocorrelation, R_h')
ax2.legend()
ax2.set_xlabel('Lag') 
plt.savefig('Q1a-3.pdf')


dFrame = pd.DataFrame(z_mat)
scatter_matrix(dFrame, figsize = (5,5))

plt.savefig('Q1a-4.pdf')

fig3, ax3 = plt.subplots(figsize = (5,5))
Contour_plot = ax3.contourf(x_grid_vec, y_grid_vec, Cov_mat)
cbar = fig3.colorbar(Contour_plot)
cbar.ax.set_ylabel('Covariance field of posterior')

plt.savefig('Q1a-5.pdf')


fig4, ax4 = plt.subplots(figsize = (5,5))
Contour_plot2 = ax4.contourf(x_grid_vec, y_grid_vec, Cov_init_mat)
cbar = fig4.colorbar(Contour_plot2)
cbar.ax.set_ylabel('Covariance field of prior')

plt.savefig('Q1a-6.pdf')

plt.tight_layout()
plt.show()


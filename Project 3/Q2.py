import pdb
import numpy as np 
import scipy as sp
import Sub_data as Sub
import pandas as pd
from scipy.stats import multivariate_normal, gamma
import matplotlib.pyplot as plt
from numpy.linalg import inv
from pandas.plotting import scatter_matrix

## Variable Definition
ktrue, xgrid, xobserved, Uobserved = Sub.load_data()

num_deg = 10                # Value of d
num_ensem = 200            # Value of M

sigma_ep = (10**(-2))
F_vec = -1.0
s_vec = Sub.source(x_vec = xobserved)
rightbc_vec = 1.0
cov_ep_mat = sigma_ep**2 * np.eye(num_deg, dtype = float)

# Initializing the states
cov_Z_init = np.eye(num_deg, dtype = float)                 # Covariance of Z (initial)
mu_Z_init = np.zeros((num_deg), dtype = float)              # Mean of Z (initial)
z_init_mat = np.zeros((num_ensem, num_deg), dtype = float)
Y_var_mat = np.zeros((num_ensem, len(xobserved)), dtype = float)
K_mat = np.zeros((num_ensem, len(xobserved)), dtype = float)
Y_mat = np.zeros((num_ensem, len(xobserved)), dtype = float)

for j1 in range(num_ensem):
    z_init_mat[j1, :] = multivariate_normal.rvs(mean = mu_Z_init, cov = cov_Z_init)
    noise_vec = multivariate_normal.rvs(mean = np.zeros((len(xobserved))), cov = sigma_ep**2 * np.eye(len(xobserved)))
    Y_var_mat[j1, :] = Sub.Y_n(x = xobserved, Z = z_init_mat[j1, :], n = num_deg, muY = 1) 
    K_mat[j1, :] = np.exp(Y_var_mat[j1, :])
    Y_mat[j1, :] = Sub.diffusioneqn(xgrid = xobserved, F = F_vec, k = K_mat[j1, :], source = s_vec, rightbc = rightbc_vec) + noise_vec

z_bar = np.mean(z_init_mat, axis = 0)
z_bar = z_bar.reshape((1, len(z_bar)))

Y_bar = np.mean(Y_mat, axis = 0)
Y_bar = Y_bar.reshape((1, len(Y_bar)))

cov_z_u_mat = np.zeros((num_deg, len(xobserved)), dtype = float)

for i1 in range(num_ensem):
    cov_z_u_mat = cov_z_u_mat + np.outer((z_init_mat[i1, :] - z_bar), (Y_mat[i1, :] - Y_bar))

cov_z_u_mat = cov_z_u_mat / num_ensem

cov_Y_mat = np.cov(Y_mat, rowvar = False)

G_mat = np.inner(cov_z_u_mat, inv(cov_Y_mat))
# print(np.mean(G_mat), np.var(G_mat))

z_new_mat = np.zeros((num_ensem, num_deg), dtype = float)
Y_new_mat = np.zeros((num_ensem, len(xobserved)), dtype = float)
K_new_mat = np.zeros((num_ensem, len(xobserved)), dtype = float)

for i2 in range(num_ensem):
    z_new_mat[i2, :] = z_init_mat[i2, :] + np.inner(G_mat, (Uobserved - Y_mat[i2, :]))
    Y_new_mat[i2, :] = Sub.Y_n(x = xobserved, Z = z_new_mat[i2, :], n = num_deg, muY = 1) 
    K_new_mat[i2, :] = np.exp(Y_new_mat[i2, :])

k_mean = np.mean(K_new_mat, axis = 0)
k_std = np.sqrt(np.var(K_new_mat, axis = 0))


# Covariance field calculation
x_grid_vec, y_grid_vec = np.meshgrid(xobserved, xobserved)
Cov_mat = np.cov(np.log(K_new_mat), rowvar = False)
Cov_init_mat = np.cov(np.log(K_mat), rowvar = False)


fig, ax = plt.subplots(figsize = (5,5))
ax.plot(xgrid, np.log(ktrue), label = 'True Value', color = 'maroon')
ax.plot(xobserved, np.log(k_mean), label = 'Kalman Update - Mean', color = 'navy', marker = '^', markersize = 8)
ax.plot(xobserved, np.log(k_mean + k_std), label = 'Kalman Update - +SD', color = 'olive', linestyle = '-.')
ax.plot(xobserved, np.log(k_mean - k_std), label = 'Kalman Update - -SD', color = 'peru', linestyle = '-.' )

ax.legend(frameon = False)
ax.set_ylabel('log Permeability, ln k(x)')
ax.set_xlabel('X coordinate, x') 
ax.set_xbound(lower = xobserved[0], upper = xobserved[-1])

dFrame = pd.DataFrame(z_new_mat)
scatter_matrix(dFrame, figsize = (5,5))

fig3, ax3 = plt.subplots(figsize = (5,5))
Contour_plot = ax3.contourf(x_grid_vec, y_grid_vec, Cov_mat)
cbar = fig3.colorbar(Contour_plot)
cbar.ax.set_ylabel('Covariance field of posterior')

fig4, ax4 = plt.subplots(figsize = (5,5))
Contour_plot2 = ax4.contourf(x_grid_vec, y_grid_vec, Cov_init_mat)
cbar = fig4.colorbar(Contour_plot2)
cbar.ax.set_ylabel('Covariance field of prior')

plt.tight_layout()
plt.show()

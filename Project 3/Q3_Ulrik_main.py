import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal, gamma
import numpy.linalg as la
import Q3_Ulrik as Sub
import matplotlib.pyplot as plt

M = np.arange(25, 1000, 100)
num_M = len(M)
RMSE_mean = np.zeros(num_M)

t0 = 0
tf = 10
delta_t = 0.05
delta_tobs = 0.1
t_arr = np.arange(t0, tf, delta_tobs)
z_0 = np.array((10,-10,10))
# z_0 = multivariate_normal.rvs(mean = np.zeros(3), cov = np.eye(3))  

sigmasq_Y = 4
K = len(t_arr)

for i1 in range(num_M):
    solve = solve_ivp(Sub.f, (t0, tf), z_0 , method='RK45', t_eval=t_arr, first_step = delta_t, next_step=delta_t)
    z_star = solve.y.T
    y_star = z_star + multivariate_normal.rvs(mean = np.zeros(3), cov = np.identity(3)*sigmasq_Y, size = len(z_star))

    # meanpos, RMSE = Sub.EnKF(Sub.integrate_step, z_0, y_star, M[i1], z_star)
    zstar, meanvec, RMSE = Sub.particlefilter(T=200, M = M[i1], delta_tobs=0.1, delta_t=0.05)
    RMSE_mean[i1] = np.mean(RMSE) # np.mean(RMSE[int(K/2):])
    print(M[i1])

print(RMSE_mean)


fig, ax = plt.subplots(figsize = (5,5))
ax.plot(M, RMSE_mean)
ax.set_xlabel('Size of ensemble, M')
ax.set_ylabel('RMSE')
plt.savefig('Q3b-1.pdf')

ax1 = plt.axes(projection='3d')
ax1.plot3D(zstar [:, 0], zstar [:, 1], zstar[:, 2], 'green')
ax1.plot3D(meanvec[:, 0], meanvec[:, 1], meanvec[:, 2], 'red')
plt.savefig('Q3b-2.pdf')

plt.show()
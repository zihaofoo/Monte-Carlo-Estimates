import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal, gamma
import numpy.linalg as la

sigmasq_Y = 4
delta_tobs = 0.05

# function to be solved
def f(t, Z):
    sigma=10
    beta=8/3
    rho=28
    c1 = -sigma*Z[0] + sigma*Z[1]
    c2 = -Z[0]*Z[2] + rho* Z[0] - Z[1]
    c3 = Z[0]*Z[1] - beta* Z[2]
    return np.array([c1, c2, c3])

# RK-4 method
def rk4(f,z0, delta_t , steps):
    
    Z = np.zeros((steps+1, 3))
    Z[0] = z0
    t=np.linspace(0, int(delta_t*steps), steps)

    for k in range(steps):
        k1 = delta_t * f(0, Z[k])
        k2 = delta_t * f(0, Z[k]+delta_t*k1/2)
        k3 = delta_t * f(0,Z[k]+delta_t*k2/2)
        k4 = delta_t * f(0,Z[k]+delta_t*k3)
        k_tot = (k1+2*k2+2*k3+k4)/6
        Z[k+1] = Z[k] + k_tot
    
    return (t,Z)

def integrate_step(v_k, stepsize):
    v_kp1 = rk4(f,v_k, stepsize , 1)[1][-1]
    return v_kp1

def EnKF(phi, z0, ystar, M, zstar, dt = delta_tobs): # phi integrates a step
    
    K = len(ystar)
    # Given initial condition and ystar
    
    v = np.zeros((K, M, 3))
    v_hat = np.zeros((K, M, 3))
    
    v_hat[0] =  multivariate_normal.rvs(z0, np.identity(3) * sigmasq_Y, size=M)
    
    # mean and cov
    meanpos = np.zeros((K, 3))
    covpos = np.zeros((K, 3))
    RMSE = np.zeros(K)
    
    etamat = multivariate_normal.rvs(np.zeros(3), np.identity(3)*sigmasq_Y, size=(K, M))
    for k in range(K-1):
        
        #step1 - Forecast
        for i in range(M):
            v[k+1, i] = phi(v_hat[k, i], dt)
        # Sample covariance
        C_MC = np.cov(v[k+1], rowvar=False)
        
        
        #Step2 - Analysis
        # Kalman gain
        K_G = np.dot(C_MC, la.inv(sigmasq_Y*np.identity(3) + C_MC ) )
        
        # Assimilation
        for i in range(M):
            v_hat[k+1, i] = v[k+1, i]  + np.inner(K_G, ystar[k+1] + etamat[k+1, i] - v[k+1, i])
        # Posterior mean and covariance
        meanpos[k+1] = np.mean(v_hat[k+1], axis=0)
        covpos_kp1 = np.cov(v_hat[k+1], rowvar=False)
        
        
        RMSE[k+1] = la.norm(meanpos[k+1] - zstar[k+1])/np.sqrt(3)
    
    return meanpos, RMSE
    
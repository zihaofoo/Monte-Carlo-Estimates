import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import multivariate_normal, multinomial
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

    k1 = f(0, z0)
    k2 = f(0, z0+delta_t*k1/2)
    k3 = f(0, z0+delta_t*k2/2)
    k4 = f(0, z0+delta_t*k3)
    k_tot = (k1+2*k2+2*k3+k4)/6
    Z_next = z0 + k_tot*delta_t
    
    return Z_next


def integrate_step(v_k, stepsize):
    v_kp1 = rk4(f,v_k, stepsize , 1)
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

def particlefilter(T=None, M=None, delta_tobs=None, delta_t=None):
    
    
    t0 = 0
    tf = delta_tobs*T
    
    t_arr = np.arange(t0, tf, delta_tobs)
    z_0 = np.array((1,1,1))
    sigmasq_Y = 4

    
    solve = solve_ivp(f, (t0, tf), z_0 , method='RK45', t_eval=t_arr, first_step = delta_t)
    zstar = solve.y.T
    ystar = zstar + multivariate_normal.rvs(mean=np.zeros(3), cov=np.identity(3)*sigmasq_Y, size=len(zstar))
    
    #initialize stuff
    x = np.zeros((T, M, 3))
    w = np.ones(M)*100
    
    x[0] = multivariate_normal.rvs(np.zeros(3), np.identity(3), size=M)
    
    #loop
    for t in range(1,T):
        Phi_t = np.array([integrate_step(x[t-1, i], delta_tobs) for i in range(M) ])
        x[t] = np.array([multivariate_normal.rvs(Phi_t[i], np.identity(3)*10**(-4)) for i in range(M) ])
        w *= np.array([ multivariate_normal.pdf(ystar[t], x[t, i], 4*np.identity(3)) for i in range(M)])
        w = w / sum(w)
        
        #check ESS condition, and if so do resampling
        ESS = 1/(sum(w**2))
        
        #RMSE
        RMSE = np.zeros(T)
        
        
        if ESS>M/10:
            #draw from multinomial
            w_copy = w.copy()
            x_copy = x.copy()
            indx_dist = np.random.multinomial(M, w) #M long array of indices which sum to M.
            start=0
            for j in range(M):
                if indx_dist[j]==0:
                    continue
                
                num = indx_dist[j]
                end = start + num
                
                for l in range(start, end):
                    x_copy[:t+1,l] = x[:t+1,j]
                w_copy[start:end] = np.array([w[j] for l in range(num) ])
                start += num
            w = w_copy.copy()
            x = x_copy.copy()
        w = w / sum(w)
    #end loop
    w = w/sum(w)
    meanvec = np.zeros((T,3))
    for i in range(M):
        meanvec += w[i]*x[:,i]

    # print(np.cov(x[-1,:,: ], rowvar= False))
    RMSE_list = 0
    for t in range(T):
        RMSE_list  += la.norm(meanvec[t] - zstar[t])/np.sqrt(3)
        
    RMSE = RMSE_list/T
    return zstar, meanvec, RMSE
        


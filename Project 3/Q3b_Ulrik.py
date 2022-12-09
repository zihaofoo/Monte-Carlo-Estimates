import numpy as np 
import scipy as sp
from scipy.stats import multivariate_normal, multinomial
import matplotlib.pyplot as plt
import Sub_data as Sub
from scipy.integrate import solve_ivp
from numpy.linalg import norm

def particlefilter(T, M, delta_tobs, ystar, zstar):
    
    #initialize stuff
    x = np.zeros((T, M, 3))
    w = np.ones(M)*100
    
    x[0] = multivariate_normal.rvs(np.zeros(3), np.identity(3), size=M)
    
    #loop
    for t in range(1,T):
        Phi_t = np.array([integrate_step(x[t-1, i], delta_tobs) for i in range(M) ])
        x[t] = np.array([multivariate_normal.rvs(Phi_t[i], np.identity(3)*10**(-4)) for i in range(M) ])
        #print('before:',w)
        w *= np.array([ multivariate_normal.pdf(ystar[t], x[t, i], 4*np.identity(3)) for i in range(M)])
        #print('after', w)
        w = w / sum(w)
        
        #check ESS condition, and if so do resampling
        ESS = 1/(sum(w**2))
        
        #RMSE
        RMSE = np.zeros(T)
        
        #print(x[:,0])
        
        if ESS>M/10:
            #draw from multinomial
            w_copy = w.copy()
            x_copy = x.copy()
            indx_dist = multinomial(M, w) #M long array of indices which sum to M.
            start=0
            for j in range(M):
                if indx_dist[j]==0:
                    continue
                
                num = indx_dist[j]
                end = start + num
                #print("Timestep:", t)
                #print("Index:", j)
                #print("number:", num)
                
                #print('before', x_copy, 'end')
                for l in range(start, end):
                    x_copy[:t+1,l] = x[:t+1,j]
                #print('after', x_copy, 'end')
                #print('weight before', w_copy)
                w_copy[start:end] = np.array([w[j] for l in range(num) ])
                
                #print('weight after', w_copy)
                
                start += num
            w = w_copy.copy()
            x = x_copy.copy()
    #end loop
    w = w/sum(w)
    meanvec = np.zeros((T,3))
    for i in range(M):
        meanvec += w[i]*x[:,i]
    RMSE = norm(meanvec - zstar)/np.sqrt(3)
    return meanvec, RMSE
        
#particlefilter(100, 10, 0.01, y_star, z_star)
import scipy.io
import numpy as np
from numpy.polynomial.legendre import leggauss
import numpy.linalg as la
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal, gamma
from numpy.random import uniform
import pdb

def load_data():

    mat1 = scipy.io.loadmat('ktrue.mat')
    mat2 = scipy.io.loadmat('inferencedata.mat')
    ktrue = [float(mat1['ktrue'][i]) for i in range(len(mat1['ktrue']))]
    xgrid = [float(mat1['xgrid'][i]) for i in range(len(mat1['xgrid']))]
    xobserved = [float(mat2['xobserved'][i]) for i in range(len(mat2['xobserved']))]
    Uobserved = [float(mat2['Uobserved'][i]) for i in range(len(mat2['Uobserved']))]
    xobserved = np.append(xobserved, 1)
    Uobserved = np.append(Uobserved, 1)

    return ktrue, xgrid, xobserved, Uobserved


def diffusioneqn(xgrid, F, k, source, rightbc):
    N = len(xgrid)
    h = xgrid[N-1] - xgrid[N-2] # assuming uniform grid

    # Set up discrete system f = Au + b using second-order FD
    A = np.zeros(shape = (N-1,N-1))
    b = np.zeros(shape = (N-1,1));
    if hasattr(source, "__len__"):
        f = -source[:(N-1)]
        f = f.reshape(len(f), 1)
    else:
        f = -source * np.ones(shape = (N-1,1))

    # diagonal entries
    A = A - 2*np.diag(k[:(N-1)]) - np.diag(k[1:]) - np.diag(np.concatenate(([k[0]], k[:(N-2)])))

    # superdiagonal
    A = A + np.diag(k[:N-2],1) + np.diag(k[1:N-1],1)

    # subdiagonal
    A = A + np.diag(k[:N-2],-1) + np.diag(k[1:N-1],-1)

    A = A / (2 * np.power(h, 2))

    # Treat Neumann BC on left side
    A[0,1] = A[0,1] + k[0] / np.power(h, 2)
    b[0] = 2.0*F/h

    # Treat Dirichlet BC on right side
    b[N-2] = rightbc * (k[N-1] + k[N-2]) / (2 * np.power(h, 2))

    # Solve it: Au = f-b
    uinternal = np.linalg.solve(A, f - b)
    usolution = np.concatenate((uinternal.reshape(N-1), [rightbc]))

    return usolution

def C(x1, x2, sigmaY = np.sqrt(0.3), L = 0.3):
    return sigmaY**2 * np.exp(-(np.abs(x1 - x2) / L))

def KL(deg = 10, dropzero = True): 
    s_, w_ = leggauss(deg)
    s = s_/2+1/2
    w = w_/2
    W = np.diag(w)
    C_mat = np.array([[C(s[i], s[j]) for i in range(deg)] for j in range(deg)])
    CW = np.dot(C_mat, W)
    sqW = sqrtm(W)
    invsqW = la.inv(sqW)
    sym = sqW @ C_mat @ sqW
    l_vec, phi_mat = la.eigh(sym)
    phi_mat = phi_mat.T
    l_vec, phi_mat = np.flip(l_vec), np.flip(phi_mat, axis=0)
    if dropzero == True:
        indxs = np.where(l_vec>10**(-10))
        l_vec = l_vec[indxs].copy() #obtain negative e vals for numerical issues
        phi_mat = phi_mat[indxs].copy()
    phi_mat_new = np.array([ invsqW @ phi_mat[i] for i in range(len(phi_mat))])
    return l_vec, phi_mat_new, s, w

def Y_n(x, Z, n = 10, muY = 1):     # Expect Z to be of length n
    l_vec, phi_mat, s, w = KL(deg = n)
    summ = 0
    phi_x = np.zeros(len(x))
    for i in range(len(l_vec)):
        phi_i = np.array([1/l_vec[i]*sum([C(x_i,s[k]) * phi_mat[i,k]*w[k] for k in range(len(s))]) 
                          for x_i in x])
        summ += np.sqrt(l_vec[i]) * phi_i * Z[i]
    return muY * np.ones(len(x)) + summ

def source(x_vec, m_vec = np.array([0.2, 0.4, 0.6, 0.8], dtype=float), theta = 0.8, delta = 0.05):
    s_mat = np.zeros((len(x_vec), len(m_vec)), dtype=float)
    for i1 in range(len(m_vec)):
        s_mat[:, i1] = (theta / (delta * np.sqrt (2 * np.pi))) * np.exp(- (x_vec - m_vec[i1])**2 / (2 * delta**2))

    # print(np.sum(s_mat, axis = 1))
    return np.sum(s_mat, axis = 1)

def PDF_posterior(l_vec, xobserved, Uobserved, sigma_epsilon, right_bc):
    (d, ) = np.shape(l_vec)             # d = Number of stochastic dimensions, n = number of MC steps.
    N = len(xobserved)                  # N = number of x coordinate points
    s_vec = source(x_vec = xobserved)

    k_n = np.exp(Y_n(x = xobserved, Z = l_vec, n = d, muY = 1))
    u_vec = diffusioneqn(xgrid = xobserved, F = -1, k = k_n, source = s_vec, rightbc = right_bc)
    # log_prior = multivariate_normal.logpdf(l_vec, mean = np.zeros(d), cov = np.identity(d), allow_singular = False)
    # log_likelihood = multivariate_normal.logpdf(Uobserved, mean = u_vec, cov = sigma_epsilon**2 * np.identity(N), allow_singular = False)
    
    log_prior = - (np.dot(l_vec, l_vec))  / 2
    log_likelihood = - ((Uobserved - u_vec).T @ la.inv(sigma_epsilon**2 * np.eye(N)) @ (Uobserved - u_vec)) / 2
    # print(log_prior, log_prior1)
    # print(log_likelihood, log_likelihood1)

    return log_prior + log_likelihood


def PDF_posterior_infer(l_vec, xobserved, Uobserved, sigma_epsilon, right_bc, label):
    (d, ) = np.shape(l_vec)             # d = Number of stochastic dimensions, n = number of MC steps.
    N = len(xobserved)                  # N = number of x coordinate points
    s_vec = source(x_vec = xobserved)

    k_n = np.exp(Y_n(x = xobserved, Z = l_vec, n = d, muY = 1))
    u_vec = diffusioneqn(xgrid = xobserved, F = -1, k = k_n, source = s_vec, rightbc = right_bc)
    # log_prior = multivariate_normal.logpdf(l_vec, mean = np.zeros(d), cov = np.identity(d), allow_singular = False)
    # log_likelihood = multivariate_normal.logpdf(Uobserved, mean = u_vec, cov = sigma_epsilon**2 * np.identity(N), allow_singular = False)
    
    log_prior = - (np.dot(l_vec, l_vec))  / 2
    log_likelihood = - ((Uobserved[label: label+1] - u_vec[label: label+1]))**2 / (2 * sigma_epsilon**2)
    # print(log_prior, log_prior1)
    # print(log_likelihood, log_likelihood1)

    return log_prior + log_likelihood

def PDF_proposal(l_vec, z_last, z_mat, epsil = 0.1):
    (t, d) = np.shape(z_mat)  # d = Number of stochastic dimensions, t = number of MC steps.
    s_d = 2.4**2 / d
    Cov_mat = (s_d * np.cov(z_mat, rowvar = False)) + (s_d * epsil * np.eye(d))
    log_l = multivariate_normal.logpdf(l_vec, mean = z_last, cov = Cov_mat, allow_singular = False)
    log_z = multivariate_normal.logpdf(z_last, mean = l_vec, cov = Cov_mat, allow_singular = False)

    return log_z - log_l 

def adaptive_MC(num_MCMC, num_deg, xobserved, Uobserved, sigma_epsilon, right_bc, epsil = 0.01):
    # Initialization
    cov_Z_init = np.eye(num_deg, dtype = float)                 # Covariance of Z (initial)
    mu_Z_init = np.zeros((num_deg), dtype = float)              # Mean of Z (initial)
    z_mat = np.zeros((num_MCMC + 1, num_deg), dtype = float)
    z_mat[0, :] = multivariate_normal.rvs(mean = mu_Z_init, cov = cov_Z_init)
    test_vec = uniform(low = 0, high = 1, size = num_MCMC)
    Accept = 0

    # Drawing from the proposal
    for i1 in range(num_MCMC):
        (t, d) = np.shape(z_mat)  # d = Number of stochastic dimensions, t = number of MC steps.
        s_d = 2.4**2 / d
        Cov_mat = (s_d * np.cov(z_mat[0:i1+1, :], rowvar = False)) + (s_d * epsil * np.eye(d))
        l_vec = multivariate_normal.rvs(mean = z_mat[i1, :], cov = Cov_mat)

        pi_log_l = PDF_posterior(l_vec, xobserved, Uobserved, sigma_epsilon, right_bc)
        pi_log_z = PDF_posterior(z_mat[i1, :], xobserved, Uobserved, sigma_epsilon, right_bc)
        q_log_l_z = PDF_proposal(l_vec, z_mat[i1, :], z_mat, epsil)
        posterior_ratio = np.exp(pi_log_l - pi_log_z + q_log_l_z)
        accept_prob = np.minimum(1.0, posterior_ratio, dtype = float)
        
        if test_vec[i1] <= accept_prob:
            z_mat[i1+1, :] = l_vec
            Accept += 1
        else:
            z_mat[i1+1, :] = z_mat[i1, :]
    
    print("Acceptance probability:", Accept / num_MCMC)
    return z_mat[num_MCMC-500:num_MCMC+1, :], z_mat


def adaptive_MC_epsilon(num_MCMC, num_deg, xobserved, Uobserved, right_bc, epsil = 0.01, gamma_1 = 1E4, gamma_2 = 1):
    # Initialization

    N = len(xobserved)
    cov_Z_init = np.eye(num_deg, dtype = float)                 # Covariance of Z (initial)
    mu_Z_init = np.zeros((num_deg), dtype = float)              # Mean of Z (initial)
    z_mat = np.zeros((num_MCMC + 1, num_deg), dtype = float)
    z_mat[0, :] = multivariate_normal.rvs(mean = mu_Z_init, cov = cov_Z_init)
    inv_sigma_sq_init = gamma.rvs(a = gamma_1, size = 1, scale = 1/gamma_2)      # Initialization of sigma_^2
    sigma_sq = np.zeros((num_MCMC + 1), dtype = float)
    sigma_sq[0] = 1 / inv_sigma_sq_init
    test_vec = uniform(low = 0, high = 1, size = num_MCMC)
    Accept = 0

    # Drawing from the proposal
    for i1 in range(num_MCMC):
        sigma_epsilon = np.sqrt(sigma_sq[i1])
        (t, d) = np.shape(z_mat)  # d = Number of stochastic dimensions, t = number of MC steps.
        s_d = 2.4**2 / d
        Cov_mat = (s_d * np.cov(z_mat[0:i1+1, :], rowvar = False)) + (s_d * epsil * np.eye(d))
        l_vec = multivariate_normal.rvs(mean = z_mat[i1, :], cov = Cov_mat)

        pi_log_l = PDF_posterior(l_vec, xobserved, Uobserved, sigma_epsilon, right_bc)
        pi_log_z = PDF_posterior(z_mat[i1, :], xobserved, Uobserved, sigma_epsilon, right_bc)
        q_log_l_z = PDF_proposal(l_vec, z_mat[i1, :], z_mat, epsil)
        posterior_ratio = np.exp(pi_log_l - pi_log_z + q_log_l_z)
        accept_prob = np.minimum(1.0, posterior_ratio, dtype = float)
        
        if test_vec[i1] <= accept_prob:
            z_mat[i1+1, :] = l_vec
            Accept += 1
        else:
            z_mat[i1+1, :] = z_mat[i1, :]
    
        # Updating sigma_sq
        s_vec = source(x_vec = xobserved)
        k_n = np.exp(Y_n(x = xobserved, Z = z_mat[i1+1, :], n = d, muY = 1))
        u_vec = diffusioneqn(xgrid = xobserved, F = -1, k = k_n, source = s_vec, rightbc = right_bc)

        gamma_alpha = gamma_1 + (N / 2)
        gamma_beta = gamma_2 + (np.dot((Uobserved - u_vec), (Uobserved - u_vec)) / 2)
        inv_sigma_sq = gamma.rvs(a = gamma_alpha, size = 1, scale = 1/gamma_beta)      
        
        sigma_sq[i1 + 1] = 1 / inv_sigma_sq

    print("Acceptance probability:", Accept / num_MCMC)

    # return z_mat, sigma_sq
    return z_mat[num_MCMC-500:num_MCMC+1, :], z_mat, sigma_sq


def variance_min(num_MCMC, num_deg, xobserved, Uobserved, sigma_epsilon, right_bc, epsil = 0.01):
    # Initialization
    mean_var = np.zeros(len(xobserved) - 1)

    for k1 in range(len(xobserved) - 1):
        cov_Z_init = np.eye(num_deg, dtype = float)                 # Covariance of Z (initial)
        mu_Z_init = np.zeros((num_deg), dtype = float)              # Mean of Z (initial)
        z_mat = np.zeros((num_MCMC + 1, num_deg), dtype = float)
        z_mat[0, :] = multivariate_normal.rvs(mean = mu_Z_init, cov = cov_Z_init)
        test_vec = uniform(low = 0, high = 1, size = num_MCMC)
        Accept = 0
        burn_in = int(num_MCMC / 2)
        k_mat = np.zeros((num_MCMC, len(xobserved)))

        # Drawing from the proposal
        for i1 in range(num_MCMC):
            (t, d) = np.shape(z_mat)  # d = Number of stochastic dimensions, t = number of MC steps.
            s_d = 2.4**2 / d
            Cov_mat = (s_d * np.cov(z_mat[0:i1+1, :], rowvar = False)) + (s_d * epsil * np.eye(d))
            l_vec = multivariate_normal.rvs(mean = z_mat[i1, :], cov = Cov_mat)

            pi_log_l = PDF_posterior_infer(l_vec, xobserved, Uobserved, sigma_epsilon, right_bc, k1)
            pi_log_z = PDF_posterior_infer(z_mat[i1, :], xobserved, Uobserved, sigma_epsilon, right_bc, k1)
            q_log_l_z = PDF_proposal(l_vec, z_mat[i1, :], z_mat, epsil)
            posterior_ratio = np.exp(pi_log_l - pi_log_z + q_log_l_z)
            accept_prob = np.minimum(1.0, posterior_ratio, dtype = float)

            if test_vec[i1] <= accept_prob:
                z_mat[i1+1, :] = l_vec
                Accept += 1
            else:
                z_mat[i1+1, :] = z_mat[i1, :]

            k_mat[i1,:] = np.exp(Y_n(xobserved, z_mat[i1, :], n = num_deg))

        mean_var[k1] = np.mean(np.var(k_mat[burn_in:, :-1], axis = 0))
        print("Acceptance probability:", Accept / num_MCMC)
    return mean_var

def int_func(t, Z_vec):
    coeff_vec = np.array([10.0, 28.0, 8.0/3])
    Z_out = np.zeros(Z_vec.shape, dtype = float)
    Z_out[0] = - (coeff_vec[0] * Z_vec[0]) + (coeff_vec[0] * Z_vec[1])
    Z_out[1] = - (Z_vec[0] * Z_vec[2]) + (coeff_vec[1] * Z_vec[0]) - Z_vec[1]
    Z_out[2] = (Z_vec[0] * Z_vec[1]) - (coeff_vec[2] * Z_vec[2]) 
    return Z_out
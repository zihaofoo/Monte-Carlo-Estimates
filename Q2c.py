_author_ = 'Zi Hao Foo'
_date_ = 'Sep 2022'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


def PDF_pi_tilde(x):
    """ Returns the PDF value of x based on pi_tilde distribution"""
    return np.exp(-0.5 * x**2) * ( (np.sin(6 * x))**2 + (3 * (np.cos(x))**2 * (np.sin(4 * x))**2) + 1 ) 

def PDF_phi(x):
    """Returns the PDF value of a standard normal density"""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def I_MC_rejection(n_MC = 10000, counter_iter_max = 1000):
    C = 20                          # Scaling constant to support g(x)
    mu_phi, sigma_phi = 0, 1        # mean and standard deviation
    num_size = np.int_(counter_iter_max)

    I_vec = np.zeros(n_MC, dtype = float)
    n_vec = np.zeros(n_MC, dtype = float)

    for i2 in range(n_MC):
        u_val = np.random.uniform(low = 0.0, high = 1.0, size = num_size)
        y_val = np.random.normal(loc = mu_phi, scale = sigma_phi, size = num_size)

        accept_bool_vec = u_val * C * PDF_phi(y_val) <= PDF_pi_tilde(y_val)
        y_vec = y_val[accept_bool_vec]

        I_vec[i2] = np.sum(y_vec ** 2) / len(y_vec)

        n_vec[i2] = len(y_vec)
    return np.mean(I_vec), np.var(I_vec)

def I_MC_importance(n_MC = 10000, counter_iter_max = 1000):
    C = 20                          # Scaling constant to support g(x)
    mu_phi, sigma_phi = 0, 1        # mean and standard deviation of standard Gassuain
    num_size = np.int_(counter_iter_max)

    I_vec = np.zeros(n_MC, dtype = float)
    n_vec = np.zeros(n_MC, dtype = float)

    for i2 in range(n_MC):          # Loop over number of independent MC simulations

        y_vec = np.random.normal(loc = mu_phi, scale = sigma_phi, size = num_size)

        pi_vec = PDF_pi_tilde(y_vec)        # Pi_tilde of y_i
        phi_vec = PDF_phi(y_vec)            # phi_tilde of y_i
        weight_vec = pi_vec / phi_vec       # w_tilde of y_i

        I_vec[i2] = np.sum( (y_vec ** 2) * weight_vec) / np.sum(weight_vec)
        n_vec[i2] = len(y_vec)

    return np.mean(I_vec), np.var(I_vec)


def I_MC_combo(n_MC = 10000, counter_iter_max = 1000):
    C = 20                          # Scaling constant to support g(x)
    mu_phi, sigma_phi = 0, 1        # mean and standard deviation
    num_size = np.int_(counter_iter_max)

    I_vec = np.zeros(n_MC, dtype = float)
    n_vec = np.zeros(n_MC, dtype = float)

    for i2 in range(n_MC):
        u_val = np.random.uniform(low = 0.0, high = 1.0, size = num_size)
        y_val = np.random.normal(loc = mu_phi, scale = sigma_phi, size = num_size)

        accept_bool_vec = u_val * C * PDF_phi(y_val) <= PDF_pi_tilde(y_val)
        y_vec = y_val[accept_bool_vec]          # Accepted samples 
        y_rej = y_val[~accept_bool_vec]         # Rejected samples

        I_AR = np.sum(y_vec ** 2) / len(y_vec)

        pi_vec = PDF_pi_tilde(y_rej)                                # Pi_tilde of y_i
        phi_prime_vec = C * PDF_phi(y_rej) - PDF_pi_tilde(y_rej)            # phi_tilde of y_i
        weight_vec = pi_vec / phi_prime_vec                           # w_tilde of y_i

        I_rej = np.sum( (y_rej ** 2) * weight_vec) / np.sum(weight_vec)

        I_vec[i2] = ((len(y_vec) / counter_iter_max) * I_AR) + ((1 - (len(y_vec) / counter_iter_max)) * I_rej)
        n_vec[i2] = len(y_vec) + len(y_rej)

    return np.mean(I_vec), np.var(I_vec)

def Int_pi(x):
    """ Returns the value of x^2 * PDF value of x based on pi_tilde distribution"""
    return (x ** 2) * PDF_pi_tilde(x)

## Integration results from quadratures
I_numer_unnorm, _ = quad(Int_pi, a = -np.inf, b = np.inf) 
I_numer_denorm, _ = quad(PDF_pi_tilde, a = -np.inf, b = np.inf) 
I_numerics = I_numer_unnorm / I_numer_denorm

print('Numerical estimate of I: ', I_numerics)

n_MC_input = 1
t_pow_vec = np.arange(start = 1, stop = 6, step = 0.25)
t_vec = np.floor(10 ** t_pow_vec)
# t_vec = np.linspace(start = 10, stop = 10000, num = 20)
I_var_rej = np.zeros(len(t_vec), dtype = float)
I_var_imp = np.zeros(len(t_vec), dtype = float)
I_var_combo = np.zeros(len(t_vec), dtype = float)
I_mean_rej = np.zeros(len(t_vec), dtype = float)
I_mean_imp = np.zeros(len(t_vec), dtype = float)
I_mean_combo = np.zeros(len(t_vec), dtype = float)

for i1 in range(len(t_vec)):
    I_mean_rej[i1], I_var_rej[i1] = I_MC_rejection(n_MC = n_MC_input, counter_iter_max = t_vec[i1]) 
    I_mean_imp[i1], I_var_imp[i1] = I_MC_importance(n_MC = n_MC_input, counter_iter_max = t_vec[i1])
    I_mean_combo[i1], I_var_combo[i1] = I_MC_combo(n_MC = n_MC_input, counter_iter_max = t_vec[i1])

# I_var_rej = I_var_rej / I_var_rej[-1]
# I_var_imp = I_var_imp / I_var_imp[-1]
# I_var_combo = I_var_combo / I_var_combo[-1]

fig, ax = plt.subplots(figsize = (10,10))
# ax.loglog(t_vec, np.abs(I_mean_rej - I_numerics)**2, linestyle = '-', linewidth = 1.5, color = 'maroon', label = 'Rejection Sampling')
ax.loglog(t_vec, np.abs(I_mean_imp  - I_numerics)**2, linestyle = '-', linewidth = 1.5, color = 'navy', label = 'Importance Sampling')
# ax.loglog(t_vec, np.abs(I_mean_combo  - I_numerics)**2, linestyle = '-', linewidth = 1.5, color = 'darkgreen', label = 'Combo')
ax.loglog([t_vec[0], t_vec[-1]], [0.5E1 / t_vec[0], 1E1 / t_vec[-1] ], linestyle = '-.', linewidth = 1.5, color = 'green', label = 'Linear Convergence' )
ax.loglog([t_vec[0], t_vec[-1]], [5E1 / t_vec[0]**2, 5E1 / t_vec[-1]**2 ], linestyle = '-.', linewidth = 1.5, color = 'red', label = 'Quadratric Convergence' )
ax.set_xlabel('Number of Monte-Carlo samples, $t$', fontsize = 18)
ax.set_ylabel('Absolute error in estimator, $ I_{MC} - I_{Quad}$', fontsize = 18)
# ax.set_ylabel('Normalized Variances, $ Var[I(t)] / Var [I(t = t_{max} )]$', fontsize = 12)
# ax.set_ylabel('Variances, $ Var[I(t)] $', fontsize = 12)
ax.legend()
# plt.savefig('Q2d_fig1.eps')
plt.show()

# fig, ax = plt.subplots(figsize = (10,10))
# ax.loglog(t_vec, I_var_rej, linestyle = '-', linewidth = 1.5, color = 'maroon', label = 'Rejection Sampling')
# ax.loglog(t_vec, I_var_imp, linestyle = '-', linewidth = 1.5, color = 'navy', label = 'Importance Sampling')
# ax.loglog(t_vec, I_var_combo, linestyle = '-', linewidth = 1.5, color = 'darkgreen', label = 'Combo')
# ax.set_xlabel('Number of Monte-Carlo samples, $t$', fontsize = 18)
# # ax.set_ylabel('Absolute error in estimator, $ I_{MC} - I_{Quad}$', fontsize = 12)
# # ax.set_ylabel('Normalized Variances, $ Var[I(t)] / Var [I(t = t_{max} )]$', fontsize = 18)
# ax.set_ylabel('Variances, $ Var[I(t)] $', fontsize = 12)
# ax.legend(loc = 'upper right', fontsize = 16, frameon = False)
# plt.savefig('Q2d_fig1.eps')
# plt.show()

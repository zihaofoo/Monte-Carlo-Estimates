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

def Int_pi(x):
    """ Returns the value of x^2 * PDF value of x based on pi_tilde distribution"""
    return (x ** 2) * PDF_pi_tilde(x)

## Integration results from quadratures
I_numer_unnorm, _ = quad(Int_pi, a = -np.inf, b = np.inf) 
I_numer_denorm, _ = quad(PDF_pi_tilde, a = -np.inf, b = np.inf) 
I_numerics = I_numer_unnorm / I_numer_denorm

print('Numerical estimate of I: ', I_numerics)

## Code for Q2b
C = 20                          # Scaling constant to support g(x)
n_MC = 10000                     # Number of independent MC simulations
mu_phi, sigma_phi = 0, 1        # mean and standard deviation of standard Gassuain
counter_iter_max = 1000         # value of t
num_size = counter_iter_max

I_vec = np.zeros(n_MC, dtype = float)
n_vec = np.zeros(n_MC, dtype = float)

for i2 in range(n_MC):          # Loop over number of independent MC simulations

    y_vec = np.random.normal(loc = mu_phi, scale = sigma_phi, size = num_size)

    pi_vec = PDF_pi_tilde(y_vec)        # Pi_tilde of y_i
    phi_vec = PDF_phi(y_vec)            # phi_tilde of y_i
    weight_vec = pi_vec / phi_vec       # w_tilde of y_i

    I_vec[i2] = np.sum( (y_vec ** 2) * weight_vec) / np.sum(weight_vec)
    n_vec[i2] = len(y_vec)

print(n_vec)

print('Expected I: ', np.mean(I_vec))
print('Variance of I: ', np.var(I_vec))
print('Expected n: ', np.mean(n_vec))

# Code for n(t)
bins = np.linspace(start = 0, stop = 2*counter_iter_max, num = int(counter_iter_max))
fig, ax = plt.subplots(figsize = (6,6))
ax.hist(n_vec, bins, color = 'maroon', label = 'Monte Carlo: t =' + str(counter_iter_max) + ', C =' + str(C))
# ax.plot([3, 3], [0, num_samples], linewidth = 1.5, color = 'blue', label = 'Analytical')
ax.set_ylabel('Frequency of values, $f$', fontsize = 12)
ax.set_xlabel('Number of accepted samples, $n(t)$', fontsize = 12)
# ax.set_title('Histogram for n = ' + str(num_vec[0]))
# ax.set_ybound([0, num_samples] )
ax.set_xbound([0, 2*counter_iter_max])
ax.set_ybound([0, counter_iter_max])
plt.legend()
# plt.savefig('Q2b_histogram.eps')
plt.show()

# Code for I(t)
bins = np.linspace(start = 0.4, stop = 1.2, num = int(counter_iter_max))
fig, ax = plt.subplots(figsize = (6,6))
ax.hist(I_vec, bins, color = 'navy', label = 'Monte Carlo: t =' + str(counter_iter_max) + ', C =' + str(C))
ax.set_ylabel('Frequency of values, $f$', fontsize = 12)
ax.set_xlabel('Monte Carlo estimate, $I_{IS}^t$', fontsize = 12)
# ax.set_title('Histogram for n = ' + str(num_vec[0]))
# ax.set_ybound([0, num_samples] )
# ax.set_xbound([0, counter_iter_max])
# ax.set_ybound([0, counter_iter_max / 2])
plt.legend()
# plt.savefig('Q2b_MCestimate.eps')
plt.show()
_author_ = 'Zi Hao Foo'
_date_ = 'Sep 2022'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm


def PDF_pi_tilde(x):
    """ Returns the PDF value of x based on pi_tilde distribution"""
    return np.exp(-0.5 * x**2) * ( (np.sin(6 * x))**2 + (3 * (np.cos(x))**2 * (np.sin(4 * x))**2) + 1 ) 

def PDF_phi(x):
    """Returns the PDF value of a standard normal density"""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

## Code for Q2a
C = 20
n_MC = 1000
mu, sigma = 0, 0.1              # mean and standard deviation
num_size = 1
counter_iter = 0  
counter_iter_max = 1000

I_vec = np.zeros(n_MC, dtype = float)
n_vec = np.zeros(n_MC, dtype = float)

for i2 in range(n_MC):
    counter_accept = 0
    y_accept_vec = np.zeros(counter_iter_max, dtype = float)

    for i1 in range(counter_iter_max): 
        u_val = np.random.uniform(low = 0.0, high = 1.0, size = num_size)
        y_val = np.random.normal(loc = mu, scale = sigma, size = num_size)

        if u_val * C * PDF_phi(y_val) <= PDF_pi_tilde(y_val):
            y_accept_vec[counter_accept] = y_val
            counter_accept = counter_accept + 1

        # counter_iter = counter_iter + 1    

    y_vec = y_accept_vec[0:counter_accept]
    # print(y_vec, len(y_vec))
    I_vec[i2] = np.sum(y_vec ** 2) / len(y_vec)
    n_vec[i2] = len(y_vec)

print(I_vec)

print('Expected I: ', np.mean(I_vec))
print('Variance of I: ', np.var(I_vec))
print('Expected n: ', np.mean(n_vec))


# Code for n(t)
bins = np.linspace(start = 0, stop = counter_iter_max, num = counter_iter_max)
fig, ax = plt.subplots(figsize = (6,6))
ax.hist(n_vec, bins, color = 'maroon', label = 'Monte Carlo: t =' + str(counter_iter_max) + ', C =' + str(C))
# ax.plot([3, 3], [0, num_samples], linewidth = 1.5, color = 'blue', label = 'Analytical')
ax.set_ylabel('Frequency of values, $f$', fontsize = 12)
ax.set_xlabel('Number of accepted samples, $n(t)$', fontsize = 12)
# ax.set_title('Histogram for n = ' + str(num_vec[0]))
# ax.set_ybound([0, num_samples] )
ax.set_xbound([0, counter_iter_max])
ax.set_ybound([0, counter_iter_max / 2])
plt.legend()
# plt.savefig('Q2a_histogram.eps')
plt.show()

# Code for I(t)
# bins = np.linspace(start = 0.0, stop = 0.04, num = counter_iter_max)
# fig, ax = plt.subplots(figsize = (6,6))
# ax.hist(I_vec, bins, color = 'navy', label = 'Monte Carlo: t =' + str(counter_iter_max) + ', C =' + str(C))
# ax.set_ylabel('Frequency of values, $f$', fontsize = 12)
# ax.set_xlabel('Monte Carlo estimate, $I_{AR}^n$', fontsize = 12)
# # ax.set_title('Histogram for n = ' + str(num_vec[0]))
# # ax.set_ybound([0, num_samples] )
# # ax.set_xbound([0, counter_iter_max])
# # ax.set_ybound([0, counter_iter_max / 2])
# plt.legend()
# plt.savefig('Q2a_MCestimate.eps')
# plt.show()


## Code to visualize f_X
# x_vec = np.linspace(start = -10, stop = 10, num = 1000)
# fig, ax = plt.subplots(figsize = (5,5))
# ax.plot(x_vec, 2 * PDF_phi(x_vec), linestyle = '-.', linewidth = 1.5, color = 'navy', label = str(2) + '$\Phi(x)$')
# ax.plot(x_vec, 5 * PDF_phi(x_vec), linestyle = '-.', linewidth = 1.5, color = 'green', label = str(5) + '$\Phi(x)$')
# ax.plot(x_vec, 10 * PDF_phi(x_vec), linestyle = '-.', linewidth = 1.5, color = 'indigo', label = str(10) + '$\Phi(x)$')
# ax.plot(x_vec, 15 * PDF_phi(x_vec), linestyle = '-.', linewidth = 1.5, color = 'red', label = str(15) + '$\Phi(x)$')
# ax.plot(x_vec, C * PDF_phi(x_vec), linestyle = '-.', linewidth = 1.5, color = 'black', label = str(C) + '$\Phi(x)$')
# ax.plot(x_vec, PDF_pi_tilde(x_vec), linestyle = '-', linewidth = 1.5, color = 'maroon', label = '$\pi (x)$')
# ax.set_xlabel('$x$', fontsize = 12)
# ax.set_ylabel('$f_X (x)$', fontsize = 12)
# ax.set_xbound([-10, 10])
# ax.set_ybound([0, 10])
# ax.legend()
# plt.savefig('Q2a_fig1.pdf')
# plt.show()
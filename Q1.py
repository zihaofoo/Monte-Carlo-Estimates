_author_ = 'Zi Hao Foo'
_date_ = 'Sep 2022'

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def X_pdf(x, alpha = 1.5):
    """PDF of random variable X"""
    bool_vec = x >= 1
    X_pdf = np.zeros(x.shape, dtype = float)
    X_pdf[bool_vec] = (alpha / (x[bool_vec] ** (alpha + 1)))  
    
    return X_pdf

def X_pdf_scalar(x, alpha = 1.5):
    """PDF of random variable X""" 

    if x >= 1.0:
        return (alpha / (x ** (alpha + 1)))
    else:
        return 0  

def func_X_U(u_vec, alpha = 1.5):
    return (1 - u_vec) ** (- 1 / alpha)

alpha = 1.5
num_samples = 10000
num_MC_points = 100000
pow_vec = np.ones(num_samples, dtype = int)
# pow_vec = np.arange(start = 1, stop = 10, step = 1)
num_vec = num_MC_points ** pow_vec

exp_vec = np.zeros(num_vec.shape)

for i1 in range(num_vec.shape[0]):
    u_vec = np.random.uniform(low=0.0, high=1.0, size=num_vec[i1])
    x_vec = func_X_U(u_vec, alpha = 1.5)
    exp_vec[i1] = np.sum(x_vec) / x_vec.size


## Code to plot Q1(0)
# fig, ax = plt.subplots(figsize = (5,5))
# ax.semilogx(num_vec, exp_vec, linestyle = '-', linewidth = 1.5, color = 'maroon', label = 'Monte Carlo')
# ax.semilogx([num_vec[0], num_vec[-1]], [alpha / (alpha - 1), alpha / (alpha - 1)], linewidth = 1.5, color = 'red', label = 'Analytical', linestyle = '-.' )
# ax.set_xlabel('Number of sample points, $n$', fontsize = 12)
# ax.set_ylabel('Expected value of X, $E_X (x)$', fontsize = 12)
# ax.set_ybound(2.5, 3.5)
# ax.set_xbound(num_vec[0], num_vec[-1])
# ax.legend()
# plt.savefig('Q1a.pdf')
# plt.show()


## Code to calculate Q1(b)
print('Number of MC points: ', num_MC_points, '\n' + 'Average: ', np.average(exp_vec))
print('Variance: ', np.var(exp_vec), '\n' + 'Skewness: ', stats.skew(exp_vec), '\n' + 'Kurtosis: ', stats.kurtosis(exp_vec))


## Code to plot Q1(a)
bins = np.linspace(start = 0, stop = 5, num = 100)
fig, ax = plt.subplots(figsize = (6,6))
ax.hist(exp_vec, bins, color = 'maroon', label = 'Monte Carlo')
ax.plot([3, 3], [0, num_samples], linewidth = 1.5, color = 'blue', label = 'Analytical')
ax.set_ylabel('Frequency of values, $f$', fontsize = 12)
ax.set_xlabel('Expected value of X, $E_X (x)$', fontsize = 12)
ax.set_title('Histogram for n = ' + str(num_vec[0]))
ax.set_ybound([0, num_samples] )
plt.legend()
# plt.savefig('Q1a' + str(num_vec[0]) + '.eps')
plt.show()


## Code to visualize f_X
# x_vec = np.linspace(start=0, stop=10, num=1000)
# x_PDF_vec = X_pdf(x_vec, alpha)
# fig, ax = plt.subplots(figsize = (5,5))
# ax.plot(x_vec, x_PDF_vec, linestyle = '-', linewidth = 1.5, color = 'maroon', label = 'X')
# ax.set_xlabel('$x$', fontsize = 12)
# ax.set_ylabel('$f_X (x)$', fontsize = 12)
# ax.legend()
# plt.show()





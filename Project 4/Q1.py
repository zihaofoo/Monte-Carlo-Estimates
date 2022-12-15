
import numpy as np
import scipy as sp
from scipy.stats import lognorm
import matplotlib.pyplot as plt

M = 1000
mu_1 = np.exp(39.58)
sigma_1 = 6.34
vec_1 = lognorm.rvs(scale = (mu_1), s = (sigma_1), size = M)

print(np.mean(np.log((vec_1))))
print(np.sqrt(np.var(vec_1)))

fig, ax = plt.subplots(figsize = (5,5))
ax.hist(np.log(vec_1), bins = 100)
# ax.set_xlabel('Value of $r_p$')
# ax.set_xlabel('Value of $\Delta x$')
# ax.set_xlabel('Value of $\Delta \Psi_D$')
ax.set_xlabel('Value of $\epsilon_{mem}$')

ax.set_ylabel('Frequency of reading')

plt.savefig('Fig1d.pdf')
plt.show()


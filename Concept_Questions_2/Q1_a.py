
_author_ = 'Zi Hao Foo'
_date_ = 'Oct_2022'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pdb
from scipy.stats import bernoulli

data_mat = np.genfromtxt('baseball.txt')[1:, :]                         # Slicing to remove header
N = 90                                              # Number of at-bats
M, _ = data_mat.shape                               # M = number of batters
sample_mat = np.zeros((N, M), dtype = float)    
P_True_vec = data_mat[:, 1]
N_rep = 10000
TSE_MC = np.zeros((N_rep, 1), dtype = float)
TSE_JS = np.zeros((N_rep, 1), dtype = float)
SE_MC = np.zeros((N_rep, M), dtype = float)
SE_JS = np.zeros((N_rep, M), dtype = float)
index_player = 10
P_dist_MC = np.zeros((N_rep, 1), dtype = float)
P_dist_JS = np.zeros((N_rep, 1), dtype = float)

for i2 in range(N_rep): 
    for i1 in range(M):
        sample_mat[:, i1] = bernoulli.rvs(data_mat[i1, 1], size = N)        # Draws sample from Bernoulli distribution with given Pi

    P_MC_vec = np.sum(sample_mat, axis = 0) / N        # Column wise sum
    TSE_MC[i2] = np.sum((P_MC_vec - P_True_vec) ** 2)       
    SE_MC[i2, :] = (P_MC_vec - P_True_vec) ** 2
    P_dist_MC[i2] = P_MC_vec[index_player]

    P_bar_N = np.sum(P_MC_vec) / len(P_MC_vec)
    sigma_0_sq = P_bar_N * (1 - P_bar_N) / N
    JS_fact = ((M - 3) * sigma_0_sq) / TSE_MC[i2]

    P_JS_vec = ((1 - JS_fact) * P_MC_vec) + (JS_fact * P_True_vec)
    TSE_JS[i2] = np.sum((P_JS_vec - P_True_vec) ** 2)       
    SE_JS[i2, :] = (P_JS_vec - P_True_vec) ** 2
    P_dist_JS[i2] = P_JS_vec[index_player]

## Total Squared Error
# fig, ax = plt.subplots(figsize = (7,7))
# ax.hist(TSE_MC, bins = 100, color = 'maroon', label = 'Vanilla MC')
# ax.hist(TSE_JS, bins = 100, color = 'navy', label = 'James-Stein')
# ax.set_xlabel('Square Estimate Error', fontsize = 14)
# ax.set_ylabel('Frequency of Estimate', fontsize = 14)
# ax.set_xbound(lower = 0, upper = 0.1)
# ax.set_ybound(lower = 0, upper = 3500)
# ax.legend(loc = 'upper right', fontsize = 14)
# # plt.savefig('Q1a_total.png')
# plt.show()

fig, ax = plt.subplots(figsize = (7,7))
ax.hist(P_dist_MC, bins = 50, color = 'maroon', label = 'Vanilla MC')
ax.hist(P_dist_JS, bins = 50, color = 'navy', label = 'James-Stein')
ax.plot([P_True_vec[index_player], P_True_vec[index_player]], [0, 3000], linestyle = 'dashed', linewidth = 2, color = 'black')
ax.set_xlabel('Square Estimate Error', fontsize = 14)
ax.set_ylabel('Frequency of Estimate', fontsize = 14)
ax.set_title('Player ' + str(index_player))
# ax.set_xbound(lower = 0, upper = 0.1)
# ax.set_ybound(lower = 0, upper = 3500)
ax.legend(loc = 'upper right', fontsize = 14)
plt.savefig('Player ' + str(index_player)+ '.png')
plt.show()


# MSE_MC = np.mean(SE_MC, axis = 0)
# MSE_JS = np.mean(SE_JS, axis = 0)
# x_index = np.arange(start = 1, stop = M + 1, step = 1)
# 
# fig, ax = plt.subplots(figsize = (8,8))
# ax.plot(x_index, MSE_MC, color = 'maroon', label = 'Vanilla MC', marker = '^', markersize = 12)
# ax.plot(x_index, MSE_JS, color = 'navy', label = 'James-Stein', marker = 'v', markersize = 12)
# ax.set_xlabel('Player Number', fontsize = 14)
# ax.set_ylabel('Average Signed Bias', fontsize = 14)
# ax.set_xticks(x_index)
# ax.set_xbound(lower = 1, upper = M)
# # ax.set_ybound(lower = 0, upper = 0.003)
# ax.legend(loc = 'upper right', fontsize = 14)
# # plt.savefig('Q1b_bias.png')
# 
# plt.show()

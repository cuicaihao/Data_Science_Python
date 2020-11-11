#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Created on   :2020/11/12 10:06:44
@author      :Caihao (Chris) Cui
@file        :Metropolis_Hastings Sampling MCMC
@content     :Beta Distribution
@version     :0.0
reference: 
https://zhuanlan.zhihu.com/p/253784711 
'''


import random
import math
from scipy.stats import beta
from scipy.stats import norm
import matplotlib.pyplot as plt

a, b = 2.31, 0.627


def norm_dist_prob(theta):
    y = beta(2.31, 0.627).pdf(theta)
    return y


T = 5000
pi = [0 for i in range(T)]
sigma = 1
t = 0
while t < T-1:
    t = t + 1
    pi_star = norm.rvs(loc=pi[t - 1], scale=sigma,
                       size=1, random_state=None)  # 状态转移进行随机抽样
    alpha = min(
        1, (norm_dist_prob(pi_star[0]) / norm_dist_prob(pi[t - 1])))  # alpha值

    u = random.uniform(0, 1)
    if u < alpha:
        pi[t] = pi_star[0]
    else:
        pi[t] = pi[t - 1]


plt.scatter(pi, beta(2.31, 0.627).pdf(pi),
            label='Target Distribution', c='red')
num_bins = 50
plt.hist(pi, num_bins, density=1, facecolor='green',
         alpha=0.7, label='Samples Distribution')
plt.legend()
plt.show()

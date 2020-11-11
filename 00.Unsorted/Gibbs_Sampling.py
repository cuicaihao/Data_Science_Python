#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Created on   :2020/11/12 10:31:22
@author      :Caihao (Chris) Cui
@file        :Gibbs_Sampling.py
@content     :2 random variables
@version     :0.0
reference: 
https://zhuanlan.zhihu.com/p/253784711 
'''
# %% Import packages
import random
import math
from scipy.stats import beta
from scipy.stats import norm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
# %% Run and check the results in 2D plot

samplesource = multivariate_normal(mean=[5, -1], cov=[[1, 0.5], [0.5, 2]])


def p_ygivenx(x, m1, m2, s1, s2):
    return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt(1 - rho ** 2) * s2))


def p_xgiveny(y, m1, m2, s1, s2):
    return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt(1 - rho ** 2) * s1))


N = 5000
K = 20
x_res = []
y_res = []
z_res = []
m1 = 5
m2 = -1
s1 = 1
s2 = 2

rho = 0.5
y = m2

for i in range(N):
    for j in range(K):
        x = p_xgiveny(y, m1, m2, s1, s2)  # y给定得到x的采样
        y = p_ygivenx(x, m1, m2, s1, s2)  # x给定得到y的采样
        z = samplesource.pdf([x, y])
        x_res.append(x)
        y_res.append(y)
        z_res.append(z)

num_bins = 50
plt.scatter(x_res, norm.pdf(x_res, loc=5, scale=1),
            label='Target Distribution x', c='green')
plt.scatter(y_res, norm.pdf(y_res, loc=-1, scale=2),
            label='Target Distribution y', c='orange')
plt.hist(x_res, num_bins, density=1, facecolor='Cyan', alpha=0.5, label='x')
plt.hist(y_res, num_bins, density=1, facecolor='magenta', alpha=0.5, label='y')
plt.title('Histogram')
plt.legend()
plt.show()


# %% Check results in 3D
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(x_res, y_res, z_res, marker='.', c='#00CED1')
plt.show()

# %%

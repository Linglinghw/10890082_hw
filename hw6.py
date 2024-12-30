# -*- coding: utf-8 -*- 

# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]


# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],[0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],[0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# m1: mean of class 1
# m2: mean of class 2
m1 = np.mean(X1, axis = 0)
m2 = np.mean(X2, axis = 0)

# write you code here
cov1 = np.cov(X1, rowvar=False)
cov2 = np.cov(X2, rowvar=False)

lambdas1, V1 = myeig(cov1, symmetric=True)
lambdas2, V2 = myeig(cov2, symmetric=True)

plt.figure(dpi=288)

plt.plot(X1[:, 0], X1[:, 1], 'r.', label='Class 1')
plt.plot(X2[:, 0], X2[:, 1], 'g.', label='Class 2')

# write you code here
scale_factor = 1
plt.quiver(m1[0], m1[1], V1[0, 0], V1[1, 0], scale=scale_factor, scale_units='xy', angles='xy', color='red', label='PC1 (Class 1)')
plt.quiver(m2[0], m2[1], V2[0, 0], V2[1, 0], scale=scale_factor, scale_units='xy', angles='xy', color='green', label='PC1 (Class 2)')

plt.axis('equal')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

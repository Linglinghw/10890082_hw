# -*- coding: utf-8 -*-

#from IPython import get_ipython
#get_ipython().run_line_magic('reset', '-sf')

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
m1 = np.mean(X1, axis = 0, keepdims=1)
m2 = np.mean(X2, axis = 0, keepdims=1)

# write you code here
S1 = (X1 - m1).T @ (X1 - m1)
S2 = (X2 - m2).T @ (X2 - m2)
S_W = S1 + S2

mean_diff = (m1 - m2).T
S_B = (m1 - m2).T @ (m1 - m2)

eigvals, eigvecs = myeig(np.linalg.inv(S_W) @ S_B)

w = eigvecs[:, 0]

w = w / np.linalg.norm(w)

X1_proj = (X1 @ w).reshape(-1, 1)
X2_proj = (X2 @ w).reshape(-1, 1)

plt.figure(dpi=288)

plt.plot(X1[:, 0], X1[:,1], 'r.', label='Class 1')
plt.plot(X2[:, 0], X2[:,1], 'g.', label='Class 2')

# write you code here

plt.plot(X1_proj * w[0], X1_proj * w[1], 'ro', alpha=0.5, label='Class 1 Proj')
plt.plot(X2_proj * w[0], X2_proj * w[1], 'go', alpha=0.5, label='Class 2 Proj')

plt.axis('equal')  
plt.show()

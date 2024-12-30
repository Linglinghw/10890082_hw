# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
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

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    # if A is full rank, no lambda value is less than 1e-6 
    # append a small value to stop rank check
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V


pts = 50
x = np.linspace(-2, 2, pts)
y = np.zeros(x.shape)

# square wave
pts2 = pts // 2
y[:pts2] = -1
y[pts2:] = 1

# sort x
argidx = np.argsort(x)
x = x[argidx]
y = y[argidx]

T0 = np.max(x) - np.min(x)
f0 = 1.0 / T0
omega0 = 2.0 * np.pi * f0

# step1: generate X=[1 cos(omega0 x) cos(omega0 2x) ... cos(omega0 nx) sin(omega0 x) sin(omega0 2x) ... sin(omega0 nx)]
n = 10
cos_terms = [np.cos(k * omega0 * x) for k in range(1, n + 1)]
sin_terms = [np.sin(k * omega0 * x) for k in range(1, n + 1)]
X = np.column_stack([np.ones(len(x))] + cos_terms + sin_terms)

# step2: SVD of X => X=USV^T
U, Sigma, VT = mysvd(X)
Sigma_inv = np.linalg.pinv(Sigma)
# step3: a = U @ S^-1 @ V^T @ y
a = VT.T @ Sigma_inv @ U.T @ y
# write your code here

y_bar = X @ a
plt.plot(x, y_bar, 'g-', label='predicted values') 
plt.plot(x, y, 'b-', label='true values')
plt.xlabel('x')
plt.xlabel('y')
plt.legend()
plt.show()



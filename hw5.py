# -*- coding: utf-8 -*-

# If this script is not run under spyder IDE, comment the following two lines.
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import math
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

def row_norm_square(X):
    return np.sum(X * X, axis=1)

# gaussian weight array g=[ g_1 g_2 ... g_m ]
# g_i = exp(-0.5 * ||x_i - c||^2 / sigma^2)
def gaussian_weight(X, c, sigma=1.0):
    s = 0.5 / sigma / sigma;
    norm2 = row_norm_square(X - c)
    g = np.exp(-s * norm2)
    return g

# xt: a sample in Xt
# yt: predicted value of f(xt)
# yt = (X.T @ G(xt) @ X)^-1 @ X.T @ G(xt) @ y
def predict(X, y, Xt, sigma=1.0):
    ntest = Xt.shape[0] # number of test samples 
    yt = np.zeros(ntest)
    for xi in range(ntest):
        c = Xt[xi, :]
        g = gaussian_weight(X, c, sigma) # diagonal elements in G
        G = np.diag(g)
        w = la.pinv(X.T @ G @ X) @ X.T @ G @ y
        yt[xi] = c @ w
    return yt

# Xs: m x n matrix; 
# m: pieces of sample
# K: m x m kernel matrix
# K[i,j] = exp(-c(|xt_i|^2 + |xs_j|^2 -2(xt_i)^T @ xs_j)) where c = 0.5 / sigma^2
# 更多實作說明, 參考課程oneonte筆記

def calc_gaussian_kernel(Xt, Xs, sigma=1):
    nt, _ = Xt.shape # pieces of Xt
    ns, _ = Xs.shape # pieces of Xs
    
    norm_square = row_norm_square(Xt)
    F = np.tile(norm_square, (ns, 1)).T
    
    norm_square = row_norm_square(Xs)
    G = np.tile(norm_square, (nt, 1))
    
    E = F + G - 2.0 * Xt @ Xs.T
    s = 0.5 / (sigma * sigma)
    K = np.exp(-s * E)
    return K

# n: degree of polynomial
# generate X=[1 x x^2 x^3 ... x^n]
# m: pieces(rows) of data(X)
# X is a m x (n+1) matrix
def poly_data_matrix(x: np.ndarray, n: int):
    m = x.shape[0]
    X = np.zeros((m, n + 1))
    X[:, 0] = 1.0
    for deg in range(1, n + 1):
        X[:, deg] = X[:, deg - 1] * x
    return X

hw5_csv = pd.read_csv('hw5.csv')
hw5_dataset = hw5_csv.to_numpy(dtype = np.float64)

hours = hw5_dataset[:, 0]
sulfate = hw5_dataset[:, 1]

# 1: 硫酸鹽濃度 vs 時間 作圖
plt.figure(figsize=(10, 5))
plt.scatter(hours, sulfate, color='red', label='Data Points')
plt.title('concentration vs time')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.legend()
plt.show()

# 2: 找迴歸方法並繪製預測曲線
# 採用13階多項式迴歸，從結果看模型在數據範圍內擬合良好，捕捉硫酸鹽濃度隨時間下降的趨勢。
degree = 13
hours_norm = (hours - np.mean(hours)) / np.std(hours)
sulfate_norm = (sulfate - np.mean(sulfate)) / np.std(sulfate)
X_norm = poly_data_matrix(hours_norm, degree)
w_norm = np.linalg.pinv(X_norm.T @ X_norm) @ X_norm.T @ sulfate_norm
predicted_norm = X_norm @ w_norm
predicted = predicted_norm * np.std(sulfate) + np.mean(sulfate)
plt.figure(figsize=(10, 5))
plt.scatter(hours, sulfate, color='red', label='Data Points')
plt.plot(hours, predicted, color='blue', label='Polynomial Fit')
plt.title('concentration vs time')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.legend()
plt.show()

# 3: 硫酸鹽濃度對數 vs 時間對數 作圖
positive_idx = (hours > 0) & (sulfate > 0)
log_hours = np.log(hours[positive_idx])
log_sulfate = np.log(sulfate[positive_idx])

plt.figure(figsize=(10, 5))
plt.scatter(hours, sulfate, color='red', label='Data Points')
plt.xscale('log')
plt.yscale('log')
plt.title('concentration vs time (log-log scale)')
plt.xlabel('time in hours (log scale)')
plt.ylabel('sulfate concentration (times $10^{-4}$, log scale)')
plt.legend()
plt.show()

# 4 :在 log-log scale 圖表上添加迴歸曲線
# 這個迴歸模型是好的，因為它在對數尺度下很好地擬合數據，解釋濃度與時間之間的冪次關係。
positive_idx = (hours > 0) & (sulfate > 0)
log_hours = np.log(hours[positive_idx])
log_sulfate = np.log(sulfate[positive_idx])
degree = 5
log_X = poly_data_matrix(log_hours, degree)
log_w = np.linalg.pinv(log_X.T @ log_X) @ log_X.T @ log_sulfate
log_predicted = log_X @ log_w
plt.figure(figsize=(10, 5))
plt.scatter(hours, sulfate, color='red', label='Data Points')
plt.xscale('log')
plt.yscale('log')
plt.plot(np.exp(log_hours), np.exp(log_predicted), color='blue', label='Log-Log Regression Line')
plt.title('concentration vs time (log-log scale)')
plt.xlabel('time in hours (log scale)')
plt.ylabel('sulfate concentration (times $10^{-4}$, log scale)')
plt.legend()
plt.show()

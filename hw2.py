# -*- coding: utf-8 -*-

# If this script is not run under spyder IDE, comment the following two lines.
# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2

plt.rcParams['figure.dpi'] = 144 

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

# 讀取影像檔, 並保留亮度成分
img = cv2.imread('svd_demo1.jpg', cv2.IMREAD_GRAYSCALE)

# convert img to float data type
A = img.astype(dtype=np.float64)

# SVD of A
U, Sigma, V = mysvd(A)
VT = V.T

def compute_energy(X: np.ndarray):
    return np.sum(X ** 2)
    # return energy of X
    # For more details on the energy of a 2D signal, see the 
    # class notebook: 內容庫/補充說明/Energy of a 2D Signal.
    # remove pass and write your code here
    
    
# img_h and img_w are image's height and width, respectively
img_h, img_w = A.shape
# Compute SNR
keep_r = 201
rs = np.arange(1, keep_r)

# compute energy of A, and save it to variable Energy_A
energy_A = compute_energy(A)

# Decalre an array to save the energy of noise vs r.
# energy_N[r] is the energy of A - A_bar(sum of the first r components)
energy_N = np.zeros(keep_r) # energy_N[0]棄置不用

for r in rs:
    # A_bar is the sum of the first r comonents of SVD
    # A_bar is an approximation of A
    A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :] 
    Noise = A - A_bar 
    energy_N[r] = compute_energy(Noise) 

# 計算snr和作圖
# write your code here
snr = 10 * np.log10(energy_A / energy_N[1:])

plt.figure()
plt.plot(rs, snr, 'r-', label='SNR')
plt.xlabel('r')
plt.ylabel('SNR')
plt.grid(False)
plt.ylim(10, None)
plt.show()

# --------------------------
# verify that energy_N[r] equals the sum of lambda_i, i from r+1 to i=n,
# lambda_i is the eigenvalue of A^T @ A
# write your code here
lambdas, _ = myeig(A.T @ A, symmetric=True)
lambdas = np.real(lambdas)
for r in rs:
    # 計算 ‖N_r‖² = energy_N[r]
    norm_Nr_squared = energy_N[r]

    # 計算 sum of λ_i from i = r+1 to n
    sum_lambda_from_r_plus_1 = np.sum(lambdas[r:])

    # 驗證 ‖N_r‖² 是否等於 sum(λ_i, i=r+1 to n)
    assert np.isclose(norm_Nr_squared, sum_lambda_from_r_plus_1), \
        f"Mismatch at r={r}: ‖N_r‖²={norm_Nr_squared}, sum(λ_i)={sum_lambda_from_r_plus_1}"

    # 計算 ‖A̅‖² = energy_A - energy_N[r]
    norm_A_bar_squared = energy_A - energy_N[r]

    # 計算 sum of λ_i from i = 1 to r
    sum_lambda_to_r = np.sum(lambdas[:r])

    # 驗證 ‖A̅‖² 是否等於 sum(λ_i, i=1 to r)
    assert np.isclose(norm_A_bar_squared, sum_lambda_to_r), \
        f"Mismatch at r={r}: ‖A̅‖²={norm_A_bar_squared}, sum(λ_i)={sum_lambda_to_r}"
    
print("All validations passed!")

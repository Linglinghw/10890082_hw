# -*- coding: utf-8 -*-

# If this script is not run under spyder IDE, comment the following two lines.
# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

def scatter_pts_2d(x, y):
    # set plotting limits
    xmax = np.max(x)
    xmin = np.min(x)
    xgap = (xmax - xmin) * 0.2
    xmin -= xgap
    xmax += xgap

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin) * 0.2
    ymin -= ygap
    ymax += ygap 

    return xmin,xmax,ymin,ymax

dataset = pd.read_csv('hw7.csv').to_numpy(dtype = np.float64)
x = dataset[:, 0]
y = dataset[:, 1]

# parameters for our two runs of gradient descent
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])

alpha = 0.05
max_iters = 500
# cost function
#     J(w0, w1, w2, w3) = sum(y[i] - w0 - w1 * sin(w2 * x[i] + w3))^2
for _ in range(1, max_iters):
    y_pred = w[0] + w[1] * np.sin(w[2] * x + w[3])
    error = y - y_pred

    grad_w0 = -2 * np.sum(error)
    grad_w1 = -2 * np.sum(error * np.sin(w[2] * x + w[3]))
    grad_w2 = -2 * np.sum(error * w[1] * x * np.cos(w[2] * x + w[3]))
    grad_w3 = -2 * np.sum(error * w[1] * np.cos(w[2] * x + w[3]))

    gradient_of_cost = np.array([grad_w0, grad_w1, grad_w2, grad_w3])
    w =  w - alpha * gradient_of_cost
    # remove the above pass and write your code here
    # calculate gradient of cost function by using partial derivative(使用偏導數計算梯度)
    # update rule: 
    #     w =  w - alpha * gradient_of_cost

xmin,xmax,ymin,ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
delta = 1e-5
for _ in range(1, max_iters):
    gradient_of_cost = np.zeros_like(w)
    for i in range(len(w)):
        w_temp = w.copy()
        w_temp[i] += delta
        cost_plus = np.sum((y - (w_temp[0] + w_temp[1] * np.sin(w_temp[2] * x + w_temp[3])))**2)

        w_temp[i] -= 2 * delta
        cost_minus = np.sum((y - (w_temp[0] + w_temp[1] * np.sin(w_temp[2] * x + w_temp[3])))**2)

        gradient_of_cost[i] = (cost_plus - cost_minus) / (2 * delta)
    w =  w - alpha * gradient_of_cost
    # remove the above pass and write your code here
    # calculate gradient of cost function by using numeric method(使用數值法計算梯度)
    # update rule: 
    #     w =  w - alpha * gradient_of_cost
    

xt = np.linspace(xmin, xmax, 100)
yt2 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

# plot x vs y; xt vs yt1; xt vs yt2 
fig = plt.figure(dpi=288)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3)
plt.plot(xt, yt1, linewidth=4, c='b', zorder=0, label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', zorder=0, label='Numeric method')
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()

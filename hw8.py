# -*- coding: utf-8 -*-

# If this script is not run under spyder IDE, comment the following two lines.
# from IPython import get_ipython
# get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

hw8_csv = pd.read_csv('hw8.csv')
hw8_dataset = hw8_csv.to_numpy(dtype = np.float64)

X0 = hw8_dataset[:, 0:2]
y = hw8_dataset[:, 2]

# write your code here
svm_clf = SVC(kernel='rbf', C=1.0, gamma=0.1)
svm_clf.fit(X0, y)

fig = plt.figure(dpi=288)

plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.legend()
plt.show()

# write your code here
# 畫出分類邊界線及著色
x_min, x_max = X0[:, 0].min() - 1, X0[:, 0].max() + 1
y_min, y_max = X0[:, 1].min() - 1, X0[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.contour(xx, yy, Z, colors='black', levels=[0])

plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.legend()
plt.show()

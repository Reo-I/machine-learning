from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(46)


def generate_data(n_data):
    x = np.linspace(-3, 3, n_data)
    pix = np.pi * x
    y = np.sin(pix) / pix + .1 * x + np.random.normal(scale=.2, size=x.shape)
    return x, y


def admm(A, ky):
    t = z = u = t_prev = np.zeros(ky.shape)
    for _ in range(1000):
        t = A.dot(ky + z - u)
        z = np.maximum(t + u - l, 0) + np.minimum(t + u + l, 0)
        u = u + t - z
        if np.linalg.norm(t - t_prev) < 1e-4:
            return t
        t_prev = t
    return t


# hyper parameters
h = .3
l = .1

x, y = generate_data(n_data=50)

# learning
design_mat = np.exp(-(x[:, None] - x[None]) ** 2 / (2 * h ** 2))
ky = design_mat.dot(y[:, None])
A = np.linalg.inv(design_mat.T.dot(design_mat) + np.identity(y.size))
t = admm(A, ky)

print('number of parameters:', t.size)
print('number of active parameters:',
      np.where(abs(t) > abs(t).mean() * 1e-4, 1, 0).sum())

# visualization
X = np.linspace(-3, 3, 1000)
K = np.exp(-(X[:, None] - x[None]) ** 2 / (2 * h ** 2))
Y = K.dot(t)
plt.plot(X, Y, color='green')
plt.scatter(x, y, c='blue', marker='o')
plt.savefig('lecture3-h2')

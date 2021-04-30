from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility


def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise


def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))


def calc_error(x, y, c, t, l, h):
    k = calc_design_matrix(x, x, h)
    theta = np.linalg.solve(
        k.T.dot(k) + l * np.identity(len(k)),
        k.T.dot(y[:, None]))
    K = calc_design_matrix(x, c, h)
    prediction = K.dot(theta)
    return np.mean((prediction.squeeze() - t) ** 2)


def calc_cv_error(x, y, l, h, n_fold=5, index=None):
    if index is None:
        index = np.random.permutation(np.arange(len(x)))
    batch_size = int(len(x) / n_fold)
    error_sum = 0
    for i in range(n_fold):
        val_begin, val_end = i * batch_size, (i + 1) * batch_size
        train_x = np.concatenate([x[index[:val_begin]], x[index[val_end:]]])
        train_y = np.concatenate([y[index[:val_begin]], y[index[val_end:]]])
        val_x = x[index[val_begin:val_end]]
        val_y = y[index[val_begin:val_end]]
        error_sum += calc_error(train_x, train_y, val_x, val_y, l, h)
    return error_sum / n_fold


# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

# k-fold cross-validation
best_l, best_h = None, None
n_fold = 5
min_error = None
index = np.random.permutation(np.arange(sample_size))
for i in range(10):
    l = 1e-5 * 10. ** i
    for j in range(10):
        h = 1e-5 * 10. ** j
        error = calc_cv_error(x, y, l, h, n_fold=5, index=index)
        if min_error is None or error < min_error:
            min_error = error
            best_l, best_h = l, h

print('Best (l, h): ({}, {})'.format(best_l, best_h))

#
# visualization
#

# calculate design matrix
k = calc_design_matrix(x, x, best_h)

# solve the least square problem
theta = np.linalg.solve(
    k.T.dot(k) + best_l * np.identity(len(k)),
    k.T.dot(y[:, None]))

# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=5000)
K = calc_design_matrix(x, X, best_h)
prediction = K.dot(theta)

plt.clf()
plt.scatter(x, y, c='green', marker='o')
plt.plot(X, prediction)
plt.savefig('lecture2-h1.png')

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


def calcurate_params(x, l, h, eps = 1e-5):
    count = 0
    num_d = x.shape[0]
    # 初期化
    k = calc_design_matrix(x, x, h)
    theta = np.random.random_sample(num_d)
    z = np.random.random_sample(num_d)
    u = np.random.random_sample(num_d)

    while(1):
        if np.sum((theta - z)**2) < eps:
            print(count)
            return theta, z, u
        elif count >=5000:
            return None, None, None
        theta = np.linalg.inv((k.T.dot(k)+ np.eye(num_d)))\
            .dot(k.T.dot(y) + z -u)
        z = np.maximum(0, theta + u - l * np.ones(theta.shape[0])) \
            - np.maximum(0, -theta - u - l*np.ones(theta.shape[0]))
        u = u +theta -z
        count +=1

# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

# calculate design matrix
h = 0.1
l = 0.05
theta, z, u = calcurate_params(x, l, h, eps = 1e-10)

# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=5000)
K = calc_design_matrix(x, X, h)
prediction = K.dot(theta)

# visualization
plt.clf()
plt.scatter(x, y, c='green', marker='o')
plt.plot(X, prediction)
#plt.savefig('lecture2-p43.png')
plt.show()

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random


# generate 2d data
n = 600
alpha = 0.1
n1 = np.sum(np.random.rand(n, 1) < alpha)
n2 = n - n1
x1 = np.random.randn(n1, 2)*np.array([1, 3]) + np.array([2, 0])
x2 = np.random.randn(n2, 2)*np.array([1, 3]) + np.array([-2, 0])


#estimate parameters
mu1_hat = np.mean(x1, axis = 0)
mu2_hat = np.mean(x2, axis = 0)
std1, std2 = np.var(np.concatenate([x1 - mu1_hat,x2 - mu2_hat]), axis = 0)
sigma_hat_inv = 1/(std1* std2) * np.array([[std2, 0], [0, std1]])
a = sigma_hat_inv.dot(mu1_hat - mu2_hat)
b = -0.5 * ((mu1_hat.T).dot(sigma_hat_inv).dot(mu1_hat) \
- (mu2_hat.T).dot(sigma_hat_inv).dot(mu2_hat)) + np.log(n1/n2)


# visualization
X1 = np.linspace(start=-1, stop=1, num=5000)
X2 = -a[0]*X1/a[1] - b/a[1]
plt.scatter(x1[:, 0], x1[:, 1], c = "r", s = 5)
plt.scatter(x2[:, 0], x2[:, 1], c = "b", s = 5)
plt.plot(X1, X2)
plt.ylim(-10, 10)
plt.show()
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


# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)


# calculate design matrix
num_CV = 5

# solve the least square problem
h_list = [0.05, 0.3, 5]
id_list = np.random.choice(sample_size, sample_size)
l_list = [0.01, 0.3, 2.0]
theta_list = []
loss_list = []
for i in h_list:
    for j in l_list:
        loss = 0
        id_list = np.random.choice(sample_size, sample_size, replace = False)
        for k in range(num_CV):
            ind = np.ones(sample_size, dtype=bool)
            ind[id_list[int(k*(sample_size/num_CV)):int((k+1)*(sample_size/num_CV))]] = False
            #print(id_list)
            x_train = x[ind]
            x_test = x[~ind]
            y_train = y[ind]
            y_test = y[~ind]
            k = calc_design_matrix(x_train, x_train, i)
            theta = np.linalg.solve(k.T.dot(k) + j * np.identity(len(k)), k.T.dot(y_train[:, None]))
            #validation
            K = calc_design_matrix(x_train, x_test, i)
            prediction = K.dot(theta)
            #loss += 0.5 * np.sum((prediction[:, 0] - y_test)**2 ) + 0.5*j*np.sum(theta**2)
            loss += 0.5 * np.sum((prediction[:, 0] - y_test) ** 2)
            #print(np.sum(ind), int(k*(sample_size/num_CV)), int((k+1)*(sample_size/num_CV)), x_train.shape, x_test.shape)
        loss_list.append(loss/num_CV)
        k = calc_design_matrix(x, x, i)
        theta = np.linalg.solve(k.T.dot(k) + j * np.identity(len(k)), k.T.dot(y[:, None]))
        theta_list.append(theta)
#print(loss_list)
# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=5000)

# visualization
count = 0
for i in range(3):
    K = calc_design_matrix(x, X, h_list[i])
    for j in range(3):
        prediction1 = K.dot(theta_list[count])
        plt.subplot(3, 3, count+1)
        plt.scatter(x, y, c='green', marker='o', s = 6, label = str(round(loss_list[count], 3)))
        plt.plot(X, prediction1)
        plt.legend(loc = "upper right")
        #plt.title("Avg. loss:" + str(round(loss_list[count], 3)))
        count += 1
#plt.savefig('lecture2-HW2.png')
plt.show()


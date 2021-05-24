import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def build_design_mat(x1, x2, bandwidth, same=True):
    K = np.zeros((x1.shape[0], x2.shape[0]))
    if same:
        for i in range(x1.shape[0]):
            K[i, i:] = \
                np.exp(-np.sum((x1[None, i] - x2[i:, :]) ** 2, axis=-1) / (2 * bandwidth ** 2))
        K = K + K.T - np.diag(np.diag(K))
    else:
        for i in range(x1.shape[0]):
            K[i, :] = np.exp(-np.sum((x1[None, i] - x2) ** 2, axis=-1) / (2 * bandwidth ** 2))
    return K

def optimize_param(design_mat, y, regularizer):
    return np.linalg.solve(
        design_mat.T.dot(design_mat) + regularizer * np.identity(len(y)),
        design_mat.T.dot(y))

def predict(train_data, test_data, theta):
    return build_design_mat(train_data, test_data, 10., same=False).T.dot(theta)


data = loadmat('digit.mat')
train = data['X']
test = data['T']
N = test.shape[1] * test.shape[2]
p_values = np.zeros((N, test.shape[2]))
num_class = 10
results = np.zeros((num_class, num_class))
for i in range(num_class):
    other = list(range(num_class))
    other.remove(i)
    one_x = np.transpose(train[:, :, i], (1, 0))
    other_x = train[:, :, other].transpose().reshape(-1, 256)
    if i == 0:
        #yは全てにおいて最初の500は正解、後の4500は不正解
        one_y = np.ones(one_x.shape[0])
        other_y = -np.ones(other_x.shape[0])
        y = np.concatenate([one_y, other_y])
    x = np.concatenate([one_x, other_x], axis=0)
    design_mat = build_design_mat(x, x, 10., same=True)
    # print(design_mat.shape)
    theta = optimize_param(design_mat, y, regularizer=1.)
    p_values[:, i] = predict(x, np.transpose(test).reshape(-1, 256), theta)
    print("{}/10: Done".format(i + 1))

for i in range(num_class):
    predict_matrix = np.argmax(p_values[i * 200:(i + 1) * 200, :], axis=1)
    idx, counts = np.unique(predict_matrix, return_counts=True)
    results[i, idx] = counts

print(results)

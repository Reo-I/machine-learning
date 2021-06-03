import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)

def generate_data(n, c):
    x = (np.random.randn(int(n/c), c) + np.linspace(-3, 3, c)).T.reshape(-1)
    return x

def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

def optimize_param(design_mat, y, regularizer):
    return np.linalg.solve(
        design_mat.T.dot(design_mat) + regularizer * np.identity(len(y)),
        design_mat.T.dot(y))

def visualization(x, thetas):
    X = np.linspace(-5, 5, 100)
    K = calc_design_matrix(x, X, 1.)
    p_y = np.array([K.dot(thetas[i, :]) for i in range(num_classes)])
    p_y = np.where(p_y < 0., 0., p_y)
    s = np.sum(p_y, axis=0)
    colors = ["blue", "r", "green"]
    mak = [",", "o", "v"]
    for i in range(num_classes):
        plt.scatter(x[i * each_num:(i + 1) * each_num], \
                    np.zeros(each_num) - (i + 1) * 0.05, c=colors[i], s=8,
                    marker=mak[i])
        plt.scatter(X, p_y[i] / s, c=colors[i], s=8, marker=mak[i])
    plt.ylim(-0.3, 1.3)
    plt.show()


n = 90
num_classes = 3
each_num = int(n/num_classes)

x= generate_data(n, num_classes)
design_mat = calc_design_matrix(x, x, 1.)
thetas = np.zeros((num_classes, n))

# calcurate the optim theta for each class
for i in range(num_classes):
    y = np.zeros(n)
    y[i*each_num:(i+1)*each_num] = 1.
    thetas[i, :] = optimize_param(design_mat, y, 1.)

visualization(x, thetas)

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)

def data_generate(n=3000):
    x = np.zeros(n)
    u = np.random.rand(n)
    index1 = np.where((0 <= u) & (u < 1 / 8))
    x[index1] = np.sqrt(8 * u[index1])
    index2 = np.where((1 / 8 <= u) & (u < 1 / 4))
    x[index2] = 2 - np.sqrt(2 - 8 * u[index2])
    index3 = np.where((1 / 4 <= u) & (u < 1 / 2))
    x[index3] = 1 + 4 * u[index3]
    index4 = np.where((1 / 2 <= u) & (u < 3 / 4))
    x[index4] = 3 + np.sqrt(4 * u[index4] - 2)
    index5 = np.where((3 / 4 <= u) & (u <= 1))
    x[index5] = 5 - np.sqrt(4 - 4 * u[index5])

    return x

def gaussian_kernel(x, d = 1):
    if d == 1:
        return np.exp(-x**2/2)/np.sqrt(2*np.pi)
n = 3000
x_i = data_generate(n)
num_crossval = 5
idx = np.random.permutation(n)
num_eachval = int(n/num_crossval)
h_list = [0.01,0.05, 0.1,0.2,  0.5]
LCV = []
for h in h_list:
    LCV_j = 0.
    for i in range(num_crossval):
        print("h:{}, {}/{}".format(h, i+1, num_crossval))
        val_idx = idx[i*num_eachval:(i+1)*num_eachval]
        train_idx = list(set(range(n)) - set(val_idx))
        train_x = x_i[train_idx]
        val_x = x_i[val_idx]
        #calcurate the density of each point
        p = [sum(map(gaussian_kernel, (val_i - train_x) / h))\
             / (n * h) for val_i in val_x]
        LCV_j+=np.mean(np.log(p))
    LCV.append(LCV_j/num_crossval)
best_idx = np.argmax(LCV)
print("best h:{} from {}->best Log Likelyhood{}".format(h_list[best_idx], h_list, LCV[best_idx]))


fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

X = np.arange(0.0,5.,0.1)
kernel = np.array([sum(map(gaussian_kernel, (e - x_i)/h_list[best_idx]))\
                   /(n * h_list[best_idx]) for e in X])
#visualize the hist of original data and the density of predicted density
ax1.hist(x_i, density = True)
#visualize the log likelyhood for each width
ax1.plot(X, kernel, label="h = "+str(h_list[best_idx]))
ax1.legend()

ax2.plot(h_list, LCV)
plt.show()
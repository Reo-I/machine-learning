import numpy as np
import matplotlib
from scipy.stats import norm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)
def data_generate(n):
    return (np.random.randn(n)+ np.where(np.random.rand(n) > 0.3, 2., -2.))

def gausian(x, mu, sigma):
    #d = 1次元
    return np.exp(-(x - mu)**2/ (2 * sigma**2))/np.sqrt(2 * np.pi * sigma**2)

def EM_algorism(x, dim):
    num_data = x.shape[0]
    w = np.random.lognormal(3., 1., dim)
    mu = np.random.normal(0, 3., dim)
    sigma = np.random.lognormal(3., 1., dim)
    L = -float('inf')
    count = 0
    while(1):
        phi = np.array([gausian(x, mu[i], sigma[i]) for i in range(dim)]).T
        L_new = np.sum(np.log(np.sum(w * phi, axis=0)))
        if np.abs(L_new - L) < 1e-8:
            print(L_new)
            return count, mu, sigma, w
        elif count >=10000:
            print("10000回で収束していない")
            return None, None, None, None
        L = L_new
        # E step
        eta = (w * phi) / np.tile(np.sum(w * phi, axis=1)[:, np.newaxis], (1, dim))\
            .reshape(num_data, dim)
        # M step
        w = np.sum(eta, axis=0) / num_data

        mu = np.array([x.dot(eta[:, j]) for j in range(dim)]) / np.sum(eta, axis=0)
        sigma = np.sqrt(np.array([eta[:, j].T.dot((x - mu[j]) ** 2) for j in range(dim)]) \
                        / np.sum(eta, axis=0))
        count +=1

# num of gausian
gausian_num = 5
num_data = 1000
x = data_generate(num_data)
c, mu, sigma, w = EM_algorism(x,gausian_num)
print("{}回で収束：平均{}, 分散{}, ガウス分布の重み{}".format(c, mu, sigma, w))
plt.hist(x, bins = 50, density = True)
X = np.arange(-5,5,0.1)
s = np.empty(X.shape)
for j in range(gausian_num):
    Y = norm.pdf(X,mu[j],sigma[j]) * w[j]
    s +=Y
    plt.plot(X,Y,color='r')
plt.plot(X,s,color='k')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def knn(train_x, train_y, test_x, k_list):
    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    dist_matrix = np.sqrt(np.sum((train_x[None] - test_x[:, None]) ** 2,
                                 axis=2))
    sorted_index_matrix = np.argsort(dist_matrix, axis=1)
    ret_matrix = None
    for k in k_list:
        knn_label = train_y[sorted_index_matrix[:, :k]]
        label_sum_matrix = None
        for i in range(10):
            predict = np.sum(np.where(knn_label == i, 1, 0), axis=1)[:, None]
            if label_sum_matrix is None:
                label_sum_matrix = predict
            else:
                label_sum_matrix = np.concatenate([label_sum_matrix,
                                                   predict], axis=1)
        if ret_matrix is None:
            ret_matrix = np.argmax(label_sum_matrix, axis=1)[None]
        else:
            ret_matrix = np.concatenate([ret_matrix, np.argmax(
                label_sum_matrix, axis=1)[None]], axis=0)
    return ret_matrix  # ret_matrix.shape == (len(k_list), len(test_x))


data = loadmat('digit.mat')
train_x = data['X'].transpose().reshape(-1, 256)
test_x = data['T'].transpose().reshape(-1, 256)
train_y = np.array([[i]*500 for i in range(10)]).reshape(-1)
k_list = [1, 2, 3, 4, 5]

num_crossval = 5
each_num = len(train_y)//num_crossval
idx = np.random.permutation(len(train_y))
results = np.zeros((len(k_list), num_crossval))
# cross validation(5)
for i in range(num_crossval):
    print("cross_val:{}/{}".format(i+1, num_crossval))
    val_idx =  idx[i*each_num:(i+1)*each_num]
    train_idx = list(set(idx) - set(val_idx))
    X_train = train_x[train_idx]
    Y_train = train_y[train_idx]
    X_val = train_x[val_idx]
    Y_val = train_y[val_idx]
    # calcurate the acc. of every candidate of 'k' at one time
    # for one train data and one validation dataset
    mat = knn(X_train, Y_train, X_val, k_list)
    results[:, i] = np.array([np.sum(mat[j, :] == Y_val)\
                              /len(Y_val)*100.0 for j in range(len(k_list))]).T

best_idx = np.argmax(np.mean(results, axis =1))
print("each acc is {}".format(np.mean(results, axis =1)))
print("best k is {}".format(k_list[best_idx]))

mat_test = knn(train_x, train_y, test_x, [k_list[best_idx]])[0]
test_y = np.array([[i]*200 for i in range(10)]).reshape(-1)
acc_list = []
result_map = np.zeros((10, 10))
acc_list.append(np.sum(mat_test == test_y)/2000 * 100.0)
for i in range(10):
  uni, counts = np.unique(mat_test[i*200:(i+1)*200], return_counts=True)
  result_map[i, uni] = counts
print(result_map)

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from collections import Counter

raw_data_x = [[3.3, 2.3],
              [3.1, 1.7],
              [1.3, 3.3],
              [3.5, 4.6],
              [2.2, 2.8],
              [7.4, 4.6],
              [5.7, 3.5],
              [9.1, 2.5],
              [7.7, 3.4],
              [7.9, 0.4]]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
X_train = np.array(raw_data_x)
Y_train = np.array(raw_data_y)
# plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='g')
# plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='r')
# plt.show()
x = np.array([8.09, 3.36])
distances = []
for x_train in X_train:
    d = sqrt(np.sum((x_train - x) ** 2))
    distances.append(d)
distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
nearst = np.argsort(distances)
k = 6
topK_y = [Y_train[i] for i in nearst[:k]]
print(topK_y)

votes = Counter(topK_y)
predict_y = votes.most_common(1)[0][0]
print("预测结果" + str(predict_y))

# 将上述操作封装成一个函数
def KNN_classify(k, X_train, y_train, x):
    assert 1 <= k <= X_train.shape[0], "k cann't be more than total samples" # k不能超过样本个数
    assert X_train.shape[0] == y_train.shape[0], "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], "the feature number of x must be equal to X_train"

    distances = [sqrt(np.sum((x_train-x)**2)) for x_train in X_train]
    nearst = np.argsort(distances)
    topK_y = [y_train[i] for i in nearst[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]
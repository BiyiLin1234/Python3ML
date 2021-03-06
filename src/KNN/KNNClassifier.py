import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:
    def __init__(self, k):
        assert k >= 1, "k must be more than 1"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._X_train is not None, "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return y_predict

    def _predict(self, x):
        # 预测单个数
        assert x.shape[0] == self._X_train.shape[1], "the feature number of x must be equal to X_train"
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearst = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearst[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]






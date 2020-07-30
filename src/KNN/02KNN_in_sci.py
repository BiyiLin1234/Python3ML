import numpy as np
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

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
x = np.array([8.09, 3.36]) # 待预测点

KNN_classfier = KNeighborsClassifier(n_neighbors=5)
KNN_classfier.fit(X_train,Y_train)
print(KNN_classfier.predict(x.reshape(1,-1))) # 必须传入数组


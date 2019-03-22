# 第1步：数据预处理
# 导入库
import pandas as pd
import numpy as np
# 导入数据集
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values
# 将类别数据数字化
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
# 躲避虚拟变量陷阱
X = X[:, 1:]
# 拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 第2步：在训练集上训练多元数据
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# 在测试集上预测结果
y_pred = regressor.predict(X_test)

import matplotlib.pyplot as plt
X1 = range(X_test.shape[0])
plt.scatter(X1, Y_test, color='red')
plt.plot(X1, y_pred, color='blue')
plt.show()
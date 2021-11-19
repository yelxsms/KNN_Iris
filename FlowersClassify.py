import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import classification_report,confusion_matrix  # 混淆矩阵和准确率，召回率，F1评分
from sklearn.model_selection import train_test_split    # 分割
from sklearn.preprocessing import StandardScaler      # 梯度下降法收敛数据
from sklearn.neighbors import KNeighborsClassifier  # knn分类器


# 装载数据集

url="iris.csv"

names =['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset=pd.read_csv(url, names=names)


# pandas的纵向切割，将数据集分割成属性（X）和标签（Y）

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

# 划分数据集和测试集,70%训练，30%测试

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

# 梯度下降法缩放数据

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

# 训练模型

classifier=KNeighborsClassifier(n_neighbors=6)   # k取6效果感觉比书上取5效果更好
classifier.fit(X_train,y_train)

# 在测试集上进行预测

y_pred=classifier.predict(X_test)

# 混淆矩阵 准确率 召回率 F1评分 评估模型优劣

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# 计算k取1到40模型的分类误差率
error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# 绘图
plt.figure(figsize=(12,6))
plt.plot(range(1,40),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

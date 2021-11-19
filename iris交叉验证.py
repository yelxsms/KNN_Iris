from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
y=iris.target

# 分成5个近邻模型
knn=KNeighborsClassifier(n_neighbors=5)

# 10折交叉验证
scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
print(scores)
print("\n")
print(scores.mean())   # 打印平均值

# 选取最优K值
k_range=range(1,31)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    k_scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
k_scores.append(scores.mean())
print(k_scores)

plt.plot(k_range,k_scores)
plt.xlabel('The value of the k in KNN')
plt.ylabel('accuracy')
plt.show()
from sklearn import datasets  # datasets模块
from sklearn.model_selection import train_test_split  # 分离训练集和测试集数据
from sklearn.neighbors import KNeighborsClassifier  # k近邻分类器模块

loaded_data = datasets.load_iris()  # 加载鸢尾花数据
X = loaded_data.data  # x有4个属性
y = loaded_data.target  # y 有三类
print(X[:2, :])  # 打印前两个样本属性

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 测试数据占30%

knn = KNeighborsClassifier()  # k近邻分离器
knn.fit(X_train, y_train)  # fit学习函数
print(knn.predict(X_test))  # 打印预测结果
print(y_test)  # 打印真实结果
print(knn.score(X_test, y_test))  # 打印正确率


#print(loaded_data)

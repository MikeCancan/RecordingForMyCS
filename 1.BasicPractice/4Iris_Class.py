import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import linear_model
Iris = datasets.load_iris()
X = Iris.data
y = Iris.target
model = linear_model.LogisticRegression(C=100)
model.fit(X,y)
print(model.coef_)
print(model.intercept_)
y_pred = model.predict(X)
print(f"Accuracy rate is {accuracy_score(y,y_pred)}")
X_draw = X[:,[2,3]]
y_draw = Iris.target
model_draw = linear_model.LogisticRegression(C=100)
model_draw.fit(X_draw,y_draw)
#对画图的区间进行定义
h=.02
x_min,x_max =X[:,2].min()- .5 ,X[:,2].max()+.5
y_min,y_max = X[:,3].min()- .5,X[:,3].max()+ .5
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),(np.arange(y_min,y_max,h)))
z = model_draw.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)
plt.pcolormesh(xx,yy,z)
plt.scatter(X[0:50,2],X[0:50,3],c='r',marker='o')
plt.scatter(X[50:100,2],X[50:100,3],c='b',marker='x')
plt.scatter(X[100:150,2],X[100:150,3],c='y',marker='+')
plt.show()
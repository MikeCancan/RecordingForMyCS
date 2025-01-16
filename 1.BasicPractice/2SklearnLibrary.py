import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as sl
def loaddata():
    data = np.loadtxt('./data/data1.txt', delimiter=',')
    n = data.shape[1]-1
    X = data[0:,0:n]
    y = data[:,-1].reshape(-1,1)
    return X,y
X,y = loaddata()
model = sl.LinearRegression()
model.fit(X,y)
#coef_输出theta1-thetan的值，intercept_输出theta0的值
print(model.coef_)
print(model.intercept_)
plt.scatter(X,y)
y_predict = model.predict(X)
plt.plot(X,y_predict)
plt.show()
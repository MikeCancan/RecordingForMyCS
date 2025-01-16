import numpy as np
import matplotlib.pyplot as plt
def loaddata():
    data = np.loadtxt("./data/data1.txt",delimiter=",")
    n = data.shape[1]-1
    X = data[:,0:n]
    y = data[:,-1].reshape(-1,1)#最后一列默认返回一个一维数组，因此需要reshape回向量形式
    return X,y
def gradientDescent(X,y,theta,iter,alpha):
    c = np.ones(X.shape[0]).transpose()#一维数组转置为列向量进行添加
    X = np.insert(X,0,values=c,axis=1) #插入对象，位置，值，axis为0表示行
    m = X.shape[0]
    n = X.shape[1]
    costs = np.zeros(iter)
    for num in range(iter):
        for j in range(n):
            theta[j] = theta[j] + (alpha/m)*(np.sum(((y-np.dot(X,theta))*X[:,j].reshape(-1,1))))
        costs[num] = computecost(X,y,theta)
    return theta,costs
def featuranormalize(X):
    mu = np.average(X,axis=0)
    sigma = np.std(X,axis=0)
    sigma[sigma==0] = 1 #某个数据正好是平均值就会导致除法错误！
    X = (X-mu)/sigma
    return X,mu,sigma
def computecost(X,y,theta):
    """
    显示在迭代过程中的损失函数值的变化
    """
    costs = np.sum(np.power((np.dot(X,theta)-y),2))/(2*X.shape[0])
    return costs
def predict(X):#用于模型预测，看拟合程度如何
    X = (X-mu)/sigma
    c = np.ones(X.shape[0]).transpose()
    X = np.insert(X,0,values=c,axis=1)
    return np.dot(X,theta)
X,y = loaddata()
theta = np.zeros(X.shape[1]+1).reshape(-1,1)
iter = 400
alpha = 0.01
X,mu,sigma = featuranormalize(X)
theta,costs = gradientDescent(X,y,theta,iter,alpha)
func_line = theta[0]+theta[1]*X
x_axis = np.linspace(1,iter,iter)
plt.plot(x_axis,costs[0:iter])#损失函数，200的时候就差不多收敛了
print(predict([[5.734]]))
"""
plt.scatter(X,y)
func_line = theta[0]+theta[1]*X
plt.plot(X,func_line)
"""
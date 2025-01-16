import numpy as np
import keras.api.datasets.mnist as kt
import matplotlib.pyplot as plt
import sklearn.linear_model as sl
from sklearn.metrics import accuracy_score

(train_x,train_y),(test_x,test_y) = kt.load_data()
X = train_x.reshape(train_x.shape[0],-1)
X = X/255
model = sl.LogisticRegression(C=100)
model.fit(X,train_y)
pred_y = model.predict(X)
print(model.coef_)#theta[1-...]
print(model.intercept_)#Theta[0]
print(f"Accuracy in train:{accuracy_score(pred_y,train_y)}")#保证标签的样本数量数一样，否则对比会报错

#原生y数据用于和predict数据进行对比
X_test = test_x.reshape(test_x.shape[0],-1)
X_test = X_test/255
y_test_pred = model.predict(X_test)
print(f"Accuracy in test:{accuracy_score(y_test_pred,test_y)}")
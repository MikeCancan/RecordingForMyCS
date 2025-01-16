import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
boston = np.loadtxt('./data/BostonHousing.csv',delimiter=',',skiprows=1)
feature_num = boston.shape[1]-1
y = boston[:,-1].reshape(-1,1)
X = boston[:,0:feature_num]
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = linear_model.Ridge()
param = {
    "alpha":[0.01,0.02,0.03,0.1,0.5,0.8],
}
GSearch = GridSearchCV(estimator=model,param_grid=param,cv=10,scoring="neg_mean_squared_error")
GSearch.fit(X_train,y_train)#得到最好的alpha参数为0.01
print(GSearch.best_params_,GSearch.best_score_)
model = linear_model.Ridge(alpha=0.01)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
print(f"MSE={mean_squared_error(y_test,y_predict)}")

#保存模型(本质上保存的就是训练的theta)
joblib.dump(model,"House_predict_model.m")
#模型读取
load_model = joblib.load("House_predict_model.m")
load_model.predict(X_test)
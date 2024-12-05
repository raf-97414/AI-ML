from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd

BosData = pd.read_csv('BostonHousing.csv') #converts csv file to pandas dataframe 
X = BosData.iloc[:,0:11] #takes first 11 columns out of 13 features 
y = BosData.iloc[:, 13] # MEDV: Median value of owner-occupied homes in $1000s
#takes last column the 14th column add iloc[] is used to extract target and features 

# Boston Housing Dataset is a derived from information collected by 
# the US Census Service concerning housing in the area of Boston MA.

# The 11 regressors/ features are
# CRIM: Per capita crime rate by town
# ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
# INDUS: Proportion of non-retail business acres per town
# CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX: Nitric oxide concentration (parts per 10 million)
# RM: Average number of rooms per dwelling
# AGE: Proportion of owner-occupied units built prior to 1940
# DIS: Weighted distances to five Boston employment centers
# RAD: Index of accessibility to radial highways
# TAX: Full-value property tax rate per $10,000
# PTRATIO: Pupil-teacher ratio by town

# The response/ target variable is
# MEDV: Median value of owner-occupied homes in $1000s


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model_used = LinearRegression()
model_used.fit(X_train,y_train)

y_train_pred = model_used.predict(X_train)
mse= mean_squared_error(y_train, y_train_pred)
r2= r2_score(y_train, y_train_pred)

print('Train MSE =', mse)
print('Train R2 score =', r2)
print("\n")

ytestpredict = model_used.predict(X_test)
mse = mean_squared_error(y_test, ytestpredict)
r2 = r2_score(y_test, ytestpredict)
print('Test MSE =', mse)
print('Test R2 score =', r2)


import matplotlib.pyplot as plt

plt.figure()
plt.scatter(y_test, ytestpredict, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Values') 
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.grid()
plt.show()
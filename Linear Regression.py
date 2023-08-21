#Linear Regression  = الانحدار الخطي
#------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
#------------------------------------

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30)
reg = LinearRegression()
f = reg.fit(x_train,y_train)
b = reg.intercept_
a = reg.coef_
y_pred = reg.predict(x_test)
#------------------------------------

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_test,reg.predict(x_test) , color='blue')
plt.title("Salary VS Experience (Training set)")
plt.xlabel("Years of experince")
plt.ylabel("Salary")
plt.show()

mse = mean_squared_error(y_test,y_pred)

Hazem = reg.predict([[3.9]])

print("Exepicted Salary = ",Hazem)


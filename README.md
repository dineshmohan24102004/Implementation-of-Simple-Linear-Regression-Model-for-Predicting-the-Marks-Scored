AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.
Equipments Required:

    Hardware – PCs
    Anaconda – Python 3.7 Installation / Jupyter notebook

Algorithm

    import pandas numpy and matplotlib
    Upload a file that contains the required data
    find x,y using sklearn
    Use line chart and disply the graph and print the mse, mae,rmse

Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DINESH.M
RegisterNumber:  212222040039
*/

import numpy as np # Supervised Learning
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
print(dataset.head()) # top rows read
print(dataset.tail()) # bottom rows read

X = dataset.iloc[:,:-1].values # need all rows but no need last column ( Independent var)
print(X)
Y = dataset.iloc[:,-1].values # Extracating Y values (Dependent Var)
print(Y)


from sklearn.model_selection import train_test_split # sklearn is a package
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =1/3,random_state=0)
# Common code for any ML model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# reg is a object
reg.fit(X_train,Y_train)
# Creating a model y=mx+c
Y_pred = reg.predict(X_test)# Checking the model (testing the model with inputdata only to check whether it predicts correctly)
print(Y_pred)
print(Y_test)


plt.scatter(X_train,Y_train,color ="green")
plt.plot(X_train,reg.predict(X_train),color ="red")
# draws a straight redline
plt.title('training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color ="blue")
plt.plot(X_test,reg.predict(X_test),color ="silver")
plt.title('test set(H vs S)')
plt.xlabel("Hours")





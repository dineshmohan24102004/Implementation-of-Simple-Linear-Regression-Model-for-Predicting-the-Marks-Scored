![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/b0aa5ce8-266d-46f9-8ef5-48128d85a317)AIM:

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


Developed by: DINESH.M
RegisterNumber:  212222040039


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/csv.csv')
df.head()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,-1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(ms)
print("RMSE = ",rmse)




OUTPUT:
![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/21a3330d-aa3b-49fb-8af4-66a62666333f)

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/24d4abc0-fa8a-46be-aa90-cf22ade342d4)

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/046e0f27-a409-49aa-9226-eacfbc01f144)
![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/0f36db8f-7386-4495-baa4-5723d376fc18)












RESULT:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.








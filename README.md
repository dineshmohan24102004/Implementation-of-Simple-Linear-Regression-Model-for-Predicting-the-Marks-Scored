
## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:

To write a program to predict the marks scored by a student using the simple linear regression model.


## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Developed by: DINESH.M
RegisterNumber:  212222040039

import pandas as pd
df= pd.read_csv('/content/student_scores.csv')
df.info()

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred =reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="silver")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_test,reg.predict(x_test),color="red")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()


mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)


a=np.array([[10]])
y_pred1=reg.predict(a)
print(y_pred1)
*/
```



## OUTPUT:

## df.head():

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/fb323e89-42cc-4243-a038-c2777eaa679a)

## df.tail():

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/763206a7-29a8-43cf-a68e-2aa8ca735c8c)

## Array value of X:

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/bdffaf0b-4eb2-43a7-9415-50e77386f191)

## Array value og Y:

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/9d37b43a-b318-4074-af03-be1edc90bf10)

## value of Y prediction :

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/875304fa-7dd2-4088-b84f-dd2f15e7214d)

## Array value of Y test:

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/9422084e-201c-4b23-9f70-d8cd19dd5341)

## Training test graph:

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/98e08aab-c506-460d-a205-14256c1e548c)

## Test set graph:

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/4a8338e1-d375-40f8-953d-f478d0d495c7)

## Value of MSE,MAE,RMSE:

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/6860b077-3d0b-4ad2-b4fb-2240037fc8a5)

















##  RESULT:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.








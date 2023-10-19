
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

```



## OUTPUT:
![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/21a3330d-aa3b-49fb-8af4-66a62666333f)

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/24d4abc0-fa8a-46be-aa90-cf22ade342d4)

![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/046e0f27-a409-49aa-9226-eacfbc01f144)
![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/80566a62-4d62-47f5-9949-e41701ab823b)
![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/622cf0a1-aee1-4721-b45a-5da64d32ece8)
![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/7d671b2f-ace2-4e37-8943-3f1954a8019d)
![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/a9b1bf73-c2b0-44a3-a816-f760d74826f9)
![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/d1a23ddd-3c3a-40d2-8b5b-8a0dc7bedea4)
![image](https://github.com/dineshmohan24102004/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119478475/25d9008b-93b5-4e10-838a-e501e4b4c799)















##  RESULT:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.








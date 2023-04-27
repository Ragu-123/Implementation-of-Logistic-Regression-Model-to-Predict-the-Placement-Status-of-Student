# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```1.import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset. 
5.Predict the values of array
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. 
7.Apply new unknown values
 ```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:RAGUNATH R
RegisterNumber: 212222240081

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:

![image](https://user-images.githubusercontent.com/113915622/234767986-d3b7a43b-d70c-474f-bcdb-f1d70c75c884.png)
![image](https://user-images.githubusercontent.com/113915622/234768879-73097022-bb73-4e21-8d92-95fe58a05e28.png)
![image](https://user-images.githubusercontent.com/113915622/234768177-2d5f1fb6-e8a6-49ff-b600-5d3245e31cbe.png)
![image](https://user-images.githubusercontent.com/113915622/234768192-5c5bba6c-d098-4f60-9fb8-06ac7ea33b3a.png)
![image](https://user-images.githubusercontent.com/113915622/234768203-9ddf0457-aae1-474e-bd7e-e2ca2826831d.png)
![image](https://user-images.githubusercontent.com/113915622/234768216-04b5a0bd-d864-460e-9235-407e98cd9bc8.png)
![image](https://user-images.githubusercontent.com/113915622/234768228-764a025f-3e37-4862-bde4-e96b3206c5d5.png)
![image](https://user-images.githubusercontent.com/113915622/234768259-dab5c6a5-2f36-4396-886a-d9e285042b06.png)
![image](https://user-images.githubusercontent.com/113915622/234768278-2015c60c-d6da-44a9-b2d0-680f7d566dcf.png)
![image](https://user-images.githubusercontent.com/113915622/234768296-11f7584d-3c49-4a9d-8476-8f56999cbdfa.png)
![image](https://user-images.githubusercontent.com/113915622/234768312-41a8c9b5-e687-41f5-93da-1a22ee2c72eb.png)
![image](https://user-images.githubusercontent.com/113915622/234768337-09bde42f-ff2a-4788-9af0-6e30b6e47d4e.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

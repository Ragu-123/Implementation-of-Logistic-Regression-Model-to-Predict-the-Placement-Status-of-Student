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

## Original data(first five columns):
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/f19c0c75-446d-4df6-b939-13ce9e3bd335)
## Data after dropping unwanted columns(first five):
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/5668ccb1-a77c-4b5c-be5a-3e8bc453b8e1)
## Checking the presence of null values:
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/21ea1ae9-f6fd-42df-8804-33337834c56d)
## Checking the presence of duplicated values:
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/a0b8a963-f5a4-4ddc-a5f2-e60ce5670273)
## Data after Encoding:
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/3dfbc564-0b7f-4067-ab8b-5a6f2b318d95)
## X Data:

![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/d21d8de5-447d-4256-9fd7-8f6195de815e)

## Y Data:
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/3dc319e8-6300-42c8-8134-4a328d091f8f)
## Predicted Values:
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/db0c0a3e-d621-4e6f-8ab0-e5c508a16759)
## Accuracy Score:
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/2cc2f259-f1b0-42be-b39a-6a70d37059ca)
## Confusion Matrix:
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/19ef32b0-c19a-407d-aa98-d3e4b854ad4e)
## Classification Report:
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/831bad7d-f832-4307-8fa5-db921fa22ad9)
## Predicting output from Regression Model
![image](https://github.com/Ragu-123/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113915622/78841272-159a-4b04-adcb-432fbf01bdf9)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

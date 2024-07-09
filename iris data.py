# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:28:10 2024

@author: jithi
"""
#here we choose our ml algorithm as RandomForest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

#load the dataset
file_path="C:/Users/jithi/OneDrive/Desktop/codesoft/IRIS.csv"
#reading and printing some coloums
iris_df=pd.read_csv(file_path)
print(iris_df.head())
#exploring the data
print(iris_df.describe())
#no null values
print(iris_df.isnull().sum())
print(iris_df['species'].value_counts())
x=iris_df.drop('species',axis=1)
y=iris_df['species']
#transform the data
from sklearn.preprocessing import LabelEncoder
Label_Encoder=LabelEncoder()
y=Label_Encoder.fit_transform(y)

#split the data into test and training
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
#now evaluating the model
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)
print(f"accuracy:{accuracy:.2f}")
print(classification_report(y_test,y_pred))
#make the predictions
sample = [[5.1, 3.5, 1.4, 0.2]]  # The input must be in 2D array format
prediction = model.predict(sample)
species = Label_Encoder.inverse_transform(prediction)
print(f"Predicted species: {species[0]}")

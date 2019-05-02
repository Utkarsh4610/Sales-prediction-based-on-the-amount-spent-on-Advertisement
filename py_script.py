# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 01:08:12 2019

@author: Utkarsh Kumar
"""
import pandas as pd
import seaborn as sns

data = pd.read_csv('Advertising.csv')
data.columns
data = data.drop(['Unnamed: 0'],axis=1)
data.isnull().sum()

sns.pairplot(data,x_vars=['TV','radio','newspaper'],y_vars='sales',kind='reg')

X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train,Y_train)

#Prediction
prediction = model.predict(X_test)  #Final predicton 

#Evaluation based on metric
from sklearn.metrics import mean_squared_error, r2_score  #importing evaluation metric
mean_squared_error(Y_test,prediction) # Calculating mean square error
r2_score(Y_test,prediction)  # Calculating r2 score

sns.distplot(Y_test, kde=False, color="b")
sns.distplot(prediction, kde=False, color="r")

"""Deployment codes starts here"""

from sklearn.externals import joblib
joblib.dump(model,'model.ml')



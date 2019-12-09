#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression models (both linear and non-linear) are used for predicting
a real value, like salary for example.

If your independent variable is time, then you are forecasting future
values, otherwise your model is predicting present but unknown values.
"""

"""
Simple Linear Regression

the equation for a line is y = mx + b + ϵ

y is the dependant variable (the variable that model will predict)

x is the independant variable

m is the coefficient or the slope of the line or how much one unit 
change in x affect a unit change in y (the greater the slope is 
the greater the change of x affects y)

b is a constant or the point that crosses the vertical axis

ϵ is the error term which is
the vertical distance between actual value and corresponding predicted value

Note ! : in the current dataset y will be the salary feature
and x will be the experience feature
"""

"""
How to find the best line that madel you data ?
By using Ordinary Least Squares (OLS) which minimize the (sum of squared prediction errors)

sum of squared errors (SSE) is the sum of the squared differences
between each observation and its group's mean. 

in other words we try to minimize the sum of squared errors.
Prediction error is defined as the difference between actual value (Y) and predicted value (Ŷ)
of dependent variable
Minimize ---> Sum(actual value - predicted value)^2

SLR minimizes the Squared Errors (SE) to optimize the parameters of a linear model, b and m,
thereby computing the best-fit line, which is represented as follows:  
y = mx + b + ϵ.
"""

"""
1- Dataset Preprocessing Phase
"""
# Import pandas library and give it an alias (pd)
import pandas as pd
# read dataset from csv file
dataset = pd.read_csv('Salary_Data.csv')
# split the dataset into dependant (y) and independant (x) features
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
# import train_test_split class
from sklearn.cross_validation import train_test_split
# split dataset to test set and train set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3, random_state = 42)

"""
Note !
Feature scaling is not required here as the following imported class
take care of it for us.

There is no need for implementing data missing technique neither
encoding categorical features.
"""

"""
2- Training Phase
"""
# import LinearRegression class
from sklearn.linear_model import LinearRegression
# create an instance of LinearRegression with called regressor
regressor = LinearRegression()
# make regressor learns from training set (Fitting regressor to train set)
# fit() takes both independant and dependant variables as parameters
regressor.fit(x_train,y_train)

"""
3- Testing and Predicting Phase
"""
# make regressor predicts the dependant variable of test set
y_predicted = regressor.predict(x_test)

"""
Visualization
"""

# visualize the model and observed points of training set
# import matplotlib.pyplot library
import matplotlib.pyplot as plt
# plot observed points of training set with red color
plt.scatter(x_train,y_train , color ="red")
# draw the line of the model
plt.plot(x_train,regressor.predict(x_train), color="blue")
# Give the digram a title
plt.title('Salary vs Experience (Training set)')
# give x axis a label
plt.xlabel('Years of Experience')
# give y axis a label
plt.ylabel('Salary')
# automatically adjust padding of the figure
plt.tight_layout()
# export the diagram or the figure to an image
plt.savefig("Training set and the model.png")
# show the diagram
plt.show()


# visualize the model and observed points of test set
plt.scatter(x_test,y_test , color ="red")
# draw the line of the model
# Note ! drawing the line with x_test will yield us the same line
# but drawing will be with different points (You can test it youself)
plt.plot(x_train,regressor.predict(x_train), color="blue")
# uncomment the next line to see it
# plt.plot(x_test,regressor.predict(x_test), color="orange")
# Give the digram a title
plt.title('Salary vs Experience (Test set)')
# give x axis a label
plt.xlabel('Years of Experience')
# give y axis a label
plt.ylabel('Salary')
# automatically adjust padding of the figure
plt.tight_layout()
# export the diagram or the figure to an image
plt.savefig("Test set and the model.png")
# show the diagram
plt.show()

"""
Note ! :
In case of the saved figures of the matplotlib are blank.
Read the following :
    
After plt.show() is called, a new figure is created.
To deal with this, you can

1- Call plt.savefig('image.png') before you call plt.show()

2- Save the figure before you show()
by calling plt.gcf() for "get current figure",
then you can call savefig() on this Figure object at any time.
like so:
    
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('img.png')
"""










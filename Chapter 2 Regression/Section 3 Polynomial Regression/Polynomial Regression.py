#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What is Polynomial Regression ?

Remember from multiple linear regression the equation that we want 
the model to learn its coefficients and its constant was
y = b0 + b1*x1 + b2*x2 + b3*x3 + ... bn*xn
where
b1 , b2 , b3 and bn are the coefficients 
b0 is the constant
y is dependant variable
x1,x2,x3 and xn are the independant variables 

Now let's look again at these independant variables all Xs are 
to the first power their exponent is 1
In the other side of this we have Polynomial Regression
which has one or more independant variables 
that raised to the powers greater than 1

y = b0 + b1*x1 + b2*x2 + b3*x3^3 + ... bn*xn^n (note the carets here!)
so the task is still the same as linear regression where we want 
the model to learn the coefficients (x1,x2,x3) and the constant (b0)
"""

"""
Preprocessing phase
"""
# import pandas library and give it an alias
import pandas as pd
# read dataset from csv file
dataset = pd.read_csv('Position_Salaries.csv')
# here we have three features
# first column : label of postion
# secund column : level
# third column : salary
# here we won't nead the label feature 
# since the level feature can be viewed as an encoded version of it
# split dataset to dependant and independant variables
# since we used libraries to implement the models 
# make sure that independant variable set x is always a metrix
# and dependant variable y is always a vector
x = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values
# the above two lines are vectors
# so we have to make sure that x is a metrix
x = dataset.iloc[:,1:2].values

# we will not spliting the dataset into training and test sets since 
# the dataset is small and has only 10 observations

# also we don't have to implement any categorical encoding technique
# as well as implementing feature scaling since the used libraries 
# take care of it for us

# let's pretend that we don't know that the current dataset couldn't
# be modeled as linear regression and let's build a one
# import the LinearRegression class
from sklearn.linear_model import LinearRegression
# create an instance of LinearRegression called regressor
regressor = LinearRegression()
# let's fit (train) the model to the dataset
regressor.fit(x,y)

# now let's visualize the model and observed data
# import pyplot module from matplotlib library and give it an alias
import matplotlib.pyplot as plt
# let's plot the data observation
plt.scatter(x,y,color="red")
# draw the linear model
plt.plot(x,regressor.predict(x),color="blue")
# give the figure a title
plt.title("Level vs Salary (Linear regression)")
# write the labels of axises
plt.xlabel("Level")
plt.ylabel("Salary")
# adjust the padding of the figure
plt.tight_layout()
# export the figure to an image
plt.savefig("Level vs Salary (Linear regression).png")
# show the figure
plt.show()

# as you can see the observations is making a curve which can be
# modeled in any way as a straight line

# let's try to build a polynomial regressor
# we will use the LinearRegression class
# but we will add the ploynomial features to the model
# import PolynomialFeatures from preprocessing module in sklearn
from sklearn.preprocessing import PolynomialFeatures
# create instance of PolynomialFeatures with degree 5
poly = PolynomialFeatures(degree = 5)
# fit x_poly to x and then transform it 
x_poly = poly.fit_transform(x)
# what this class actually do is Generate a new feature matrix
# consisting of all polynomial combinations of the features 
# with degree less than or equal to the specified degree.
# so x_poly will contains x^0 , x^1 , x^2 and x^3 columns
# and with this done let's give the linear regressor object
# the new metrix of independant variables x_poly
# let's retrain the model on this x_poly
regressor.fit(x_poly,y)
# now let's visualize the new model with the observations
# create a new figure (so matplotlib won't draw figures
# above each other)
plt.figure()
# plot the data observation
plt.scatter(x,y,color="red")
# draw the linear model that predict the new x_poly
plt.plot(x,regressor.predict(x_poly),color="blue")
# give the figure a title
plt.title("Level vs Salary (Polynomial regression)")
# write the labels of axises
plt.xlabel("Level")
plt.ylabel("Salary")
# adjust the padding of the figure
plt.tight_layout()
# export the figure to an image
plt.savefig("Level vs Salary (Polynomial regression).png")
# show the figure
plt.show()

# let's make the curve smoother
# how this being done ? 
# by ploting more points that belongs to the curve

# let's create a list of values that starts from the minimum of level
# feature up to the maximum increased by 0.1 by using numpy library
# import numpy and give it an alias
import numpy as np
x_grid = np.arange(min(x), max(x), 0.1)
# reshape the x_grid into a matrix 
# with columns count = length of x_grid and rows count = 1
x_grid = np.reshape(x_grid,(len(x_grid),1))
# now let's visualize the new model with the observations
# create a new figure
plt.figure()
# plot the data observation
plt.scatter(x,y,color="red")
# draw the linear model that predict the new x_grid
plt.plot(x_grid,regressor.predict(poly.fit_transform(x_grid)),color="blue")
# give the figure a title
plt.title("Level vs Salary (Polynomial regression)")
# write the labels of axises
plt.xlabel("Level")
plt.ylabel("Salary")
# adjust the padding of the figure
plt.tight_layout()
# export the figure to an image
plt.savefig("Level vs Salary (Polynomial regression) smoother curve.png")
# show the figure
plt.show()

# let's try to use the linear and ploynomial regression models
# to predict the salary at level = 6.5

# train both models
linearRegressor = LinearRegression()
# train linearRegressor
linearRegressor.fit(x,y)

ploynomialRegressor = LinearRegression()
# train ploynomialRegressor
ploynomialRegressor.fit(x_poly,y)

# linear Regression prediction
linearRegressor.predict(6.5)
# ploynomial Regression prediction
ploynomialRegressor.predict(poly.fit_transform(6.5))
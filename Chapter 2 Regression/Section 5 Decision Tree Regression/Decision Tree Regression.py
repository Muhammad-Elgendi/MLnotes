#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision Tree Regression

Decision Tree algorithm is able to build a discrete model only
all of the previous discussed algorithms help us to create a continuous
model of the data
"""

"""
What is information entropy ?
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
# the above two lines give us two vectors
# so we have to make sure that x is like a metrix
x = dataset.iloc[:,1:2].values

# we will not spliting the dataset into training and test sets since 
# the dataset is small and has only 10 observations

# also we don't have to implement any categorical encoding technique
# as well as implementing feature scaling since the used libraries 
# take care of it for us

# import the DecisionTreeRegressor class
from sklearn.tree import DecisionTreeRegressor
# create an instance of DecisionTreeRegressor called regressor
# another thing is when you excute this script multible times
# you will get different predictions each time to solve this 
# behavior we give the random_state parameter a value
regressor = DecisionTreeRegressor(random_state = 0)
# let's fit (train) the model to the dataset
regressor.fit(x,y)

# now let's visualize the model and observed data
# import pyplot module from matplotlib library and give it an alias
import matplotlib.pyplot as plt
# create a new figure with size 10*5 inch
plt.figure().set_size_inches(10, 5)
# let's plot the data observation
plt.scatter(x,y,color="red")
# draw the discrete model
# if we plot the point that are in the dataset we will get 
# a non-acurate visualization since the constructed model
# isn't continous we will draw the model in higher resolution
# which mean create much more points to plot
# we will create a range of points using np.range() method
# import numpy and give it an alias
import numpy as np
x_higher = np.arange(min(x),max(x),0.01)
# let's reshape this numpy array so it's like the original x matrix
x_higher = np.reshape(x_higher,(len(x_higher),1),1)
# now let's plot all of these new points to create an accurate 
# representation
plt.plot(x_higher,regressor.predict(x_higher),color="blue")
# uncomment the following line to see the inaccurate model
# plt.plot(x,regressor.predict(x),color="blue")

# give the figure a title
plt.title("Level vs Salary (Decision Tree Regression)")
# write the labels of axises
plt.xlabel("Level")
plt.ylabel("Salary")
# adjust the padding of the figure
plt.tight_layout()
# export the figure to an image
plt.savefig("Level vs Salary (Decision Tree Regression).png")
# show the figure
plt.show()

# let's try to use the Random Forest Regression model
# to predict the salary at level = 6.5
# predict the salary
salary = regressor.predict(6.5)
# print it to the console
print('salary for 6.5 level is : '+str(salary))



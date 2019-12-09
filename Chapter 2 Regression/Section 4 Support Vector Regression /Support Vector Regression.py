#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What is Support Vector Regression (SVR) ?
Support Vector Regression (SVR) works on similar principles as Support Vector Machine (SVM) classification. One can say that SVR is the adapted form of SVM when the dependent variable is numerical rather than categorical. A major benefit of using SVR is that it is a non-parametric technique. Unlike SLR, whose results depend on Gauss-Markov assumptions, the output model from SVR does not depend on distributions of the underlying dependent and independent variables. Instead the SVR technique depends on kernel functions. Another advantage of SVR is that it permits for construction of a non-linear model without changing the explanatory variables, helping in better interpretation of the resultant model. The basic idea behind SVR is not to care about the prediction as long as the error (ϵi) is less than certain value. This is known as the principle of maximal margin. This idea of maximal margin allows viewing SVR as a convex optimization problem. The regression can also be penalized using a cost parameter, which becomes handy to avoid over-fit. SVR is a useful technique provides the user with high flexibility in terms of distribution of underlying variables, relationship between independent and dependent variables and the control on the penalty term.


SVR technique relies on kernel functions to construct the model. The commonly used kernel functions are: a) Linear, b) Polynomial, c) Sigmoid and d) Radial Basis. While implementing SVR technique, the user needs to select the appropriate kernel function.  The selection of kernel function is a tricky and requires optimization techniques for the best selection.


Given a non-linear relation between the variables of interest and difficulty in kernel selection, we would suggest the beginners to use RBF as the default kernel. The kernel function transforms our data from non-linear space to linear space. The kernel trick allows the SVR to find a fit and then data is mapped to the original space.

We have learnt that the real value of Root Mean Square Error (RMSE) lies in the comparison of alternative models. In the SVR model, the predicted values are closer to the actual values, suggesting a lower RMSE value. RMSE calculation would allow us to compare the SVR model with the earlier constructed linear model. A lower value of RMSE for SVR model would confirm that the performance of SVR model is better than that of SLR model. 

The SVR technique is flexible in terms of maximum allowed error and penalty cost. This flexibility allows us to vary both these parameters to perform a sensitivity analysis in attempt to come up with a better model. Now we will perform sensitivity analysis, by training a lot of models with different allowable error and cost parameter. This process of searching for the best model is called tuning of SVR model.

"""

"""
How to tune the SVR model ?

by varying maximum allowable error and cost parameter.
the value of Mean Square Error (MSE). MSE is defined as (RMSE)2 and is also a performance indicator.
Tuning the model is extremely important as it optimizes the parameters for best prediction. 
"""

"""
What is the kernel ?
"""

"""
What are the types of kernels ?
"""

"""
What is kernel trick ?
"""

# import pandas library and give it an alias
import pandas as pd
# read dataset from the csv file
dataset = pd.read_csv("Position_Salaries.csv")
# here we have three features
# first column : label of postion
# secund column : level
# third column : salary
# here we won't nead the label feature 
# since the level feature can be viewed as an encoded version of it
# split dataset to dependant and independant variables
y = dataset.iloc[:,2].values
# make sure that x is considered as a metrix
x = dataset.iloc[:,1:2].values
# we will not spliting the dataset into training and test sets since 
# the dataset is small and has only 10 observations
# also we don't have to implement any categorical encoding technique

# feature scaling (SVR class didn't implement it)
# import StandardScaler class
from sklearn.preprocessing import StandardScaler
# create different scaler objects for x and y so we can 
# inverse each one
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# scale independant variable x
x = scaler_x.fit_transform(x)
# scale dependant variable y
y = scaler_y.fit_transform(y)

# build the model
# import SVR class from sklearn library
from sklearn.svm import SVR

"""
what are the loss functions ?
"""

# create regressor object
# and Specifing the kernel type to be used in the algorithm. 
# It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
# ‘precomputed’ or a callable.
# If none is given, ‘rbf’ will be used.
# let's use radial basis function kernel
regressor = SVR(kernel = 'rbf')
# train the regressor on the whole dataset
regressor.fit(x,y)

# visulize the model and dataset observarions
# import matplotlib.pyplot library and give it an alias
import matplotlib.pyplot as plt
# plot the observations of the dataset in red color
plt.scatter(x,y, color='red')
# plot the regressor model in blue
plt.plot(x,regressor.predict(x),color="blue")
# give the figure a title
plt.title("Level vs Salary (Support Vector Regression)")
# write the labels of axises
plt.xlabel("Level")
plt.ylabel("Salary")
# adjust the padding of the figure
plt.tight_layout()
# export the figure to an image
plt.savefig("Level vs Salary (SVR).png")
# show the figure
plt.show()

"""
The model did quite well predictions except for the CEO salary
why is that ?
becasue the salary of CEO is considered as outlier

How to detect outliers in the data ?


"""

# make a prediction of salary for 6.5 level
# first you have to scale the input as we scaled the x variable
# so we use scaler_x.transform since the scaler is already fitted
# but this transform array parameter is array-like,
# in shape [n_samples, n_features] so we will create this parameter
# like so
# import numpy and give it an alias
import numpy as np
# prepare an input for the scaler
prepared_input = np.array([[6.5]])
# scale the input
scaled_input = scaler_x.transform(prepared_input)
# predict the salary for the input
# and since the model trained on scaled values, when we use it
# we also give it a scaled input
predicted_value = regressor.predict(scaled_input)
# again the output of the model is also scaled so to transform it 
# into something we can read we use inverse_transform() of y scaler
salary = scaler_y.inverse_transform(predicted_value)
# print the salary
print('salary for 6.5 level is : '+str(salary))






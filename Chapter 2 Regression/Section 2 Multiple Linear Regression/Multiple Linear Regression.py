#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What is the P-value meaning ? 
In plain english:
The p-value is actually the probability of getting a sample like ours,
or more extreme than ours IF the null hypothesis is true.

So, we assume the null hypothesis is true
and then determine how “strange” our sample really is.

If it is not that strange (a large p-value) 
then we don’t change our mind about the null hypothesis.

As the p-value gets smaller, we start wondering
if the null hypothesis really is true and well maybe we should 
change our minds (and reject the null hypothesis).

In more formal way :
The p-value is used to determine 
if the outcome of an experiment is statistically significant.

A high p-value means that, assuming the null hypothesis is true,
this outcome was very likely.

A low p-value means that, assuming the null hypothesis is true,
there is a very low likelihood that this outcome was a result of luck.
"""

"""
How to calculate p-value ?
"""

"""
what is dummy variable trap ?
"""

"""
What to do if we have multiple independant variable and one of
them is categorical ?
"""

"""
What is the techniques to select the variables that is will be in our
model ?
1- All variables in the model (Not recommended)
2- Backward elimination
3- Forward selection
4- Bidirectional elimination aka Stepwise regression
5- Score comparision
"""
# import pandas library and give it an alias (pd)
import pandas as pd
# load the dataset from csv file
dataset = pd.read_csv('50_Startups.csv')

# split dataset into independant x and dependant y variables
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# encode the categorical features of the dataset
# here we have state feature is last feature 
# so we can use either 3 or the reverse order which is -1

# we first need to apply LabelEncoder which will encode the text into
# numbers then we will apply the OneHotEncoder to create the dummy 
# variables corresponding to state feature values (which are numbers)
# because OneHotEncoder can't encode text values

# import OneHotEncoder,LabelEncoder classes
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
# create an instance of LabelEncoder class
"""
used to transform non-numerical labels 
(as long as they are hashable and comparable) to numerical labels.
"""
labelEncoder = LabelEncoder()
# apply lablel encoding to state feature
x[:,-1] = labelEncoder.fit_transform(x[:,-1])

# create an instance of OneHotEncoder class
# with parameter categorical_features which is an array
# that has the indexs of categorical data
oneHotEncoder = OneHotEncoder(categorical_features = [-1])
# apply the OneHot Encoding to independant variables x
x = oneHotEncoder.fit_transform(x).toarray()

# Avoid the dummy varible trap
# (This is optional because the LinearRegression class avoids it)
# mainly avoiding the trap is done by keeping n-1 of dummy variables
# which meaning deleting one of them (we will delete the first one)
x = x[:,1:]

# split dataset into training and test set
# import train_test_split class
from sklearn.cross_validation import train_test_split
# split the dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 ,random_state = 42)

# we will not applying any feature scaling here since the library
# take care of it for us

# import LinearRegression class
from sklearn.linear_model import LinearRegression
# create an instance of LinearRegression called regressor
regressor = LinearRegression()
# train the model with training set
regressor.fit(x_train,y_train)
# predict test set target values or independant variable values
# and assign the predicted values to y_predicted
y_predicted = regressor.predict(x_test)

# as you can see we take all the independant variables into our model
# but this is not recommanded to be done so we will find the best
# independant variables that is statisticaly significant to 
# the dependant variable

# and this can applied by one of the five techniques stated before
# but now we will choose only one of them
 
# let's apply the backward elimination

# one of the interesting libraries that will calculate p-values for
# each independant variable is statsmodel library specifically
# the summary() method

# let's import statsmodel library and give it an alias (sm)
import statsmodels.formula.api as sm

# remember the linear model equation for mulible variables
# it's y = b0 + b1*x1 + b2*x2 + b3*x3 ..... + bn*xn 
# for n of variables 

# again b0 here is a constant like b in the quation stated before
# (y = mx + b)
# and all of b1 , b2 , b3 .... and bn are a coefficient like m 
# in the above equation

# but here in the statsmodels implementation of linear equation
# the constant b0 is not exist so we add a new independant variable
# to our dataset but what value we are going to add so that the 
# model can find the best constant value without any of effects of
# our new added value
# well , as you can see we will add this (b0*x0) to the statsmodels
# linear equation and x0 here will the new added independant variable
# to the dataset
# and since b0*1 = b0 we will assign one(1) to our new added variable

# adding a column with ones to the dependant variable x
# using numpy append method
# let's import numpy library and give it an alias (np)
import numpy as np
# adding a column with ones to the dependant variable x
# and since we want the column of ones to be the first column in x
# we will append x to a vector of ones 
x = np.append(arr = np.ones((50,1),np.int) ,
              values = x ,
              axis = 1)

# now we will start backward elimination
"""
Backward Elimination 

STEP 1: Select a significance level to stay in the model
(e.g. SL = 0.05) 

STEP 2: Fit the model with all possible predictors 

STEP 3: Consider the predictor with the highest P-value.
If P > SL, go to STEP 4, otherwise go to END 

STEP 4: Remove the predictor 

STEP 5: Fit model without this variable

END: Your Model Is Ready
"""
# step 1 let's say our significance level is 0.05

# step 2 let's fit the model with all posible predictors
# let's create a variable called x_optimal that will contains
# all independant variables for now
x_optimal = x[:]

# step 3 : let's create a new model and fit it to x_optimal 
# OLS is the class of linear model in statsmodels library
sm_regressor = sm.OLS(exog = x_optimal , endog = y).fit()

# now let's see the p-value of each independant variable
sm_regressor.summary()

"""
variable   p-value

const      0.000
x1         0.953 
x2         0.990  
x3         0.000
x4         0.608 
x5         0.123
"""

# step 3 Consider the predictor with the highest P-value.
# If P > SL, go to STEP 4, otherwise go to END 
# here we consider x2 and go to step 4
# step 4 we will remove x2 
# which is the third dummy variable of state feature
# and its index is 2 from x_optimal
x_optimal = x[:,[0,1,3,4,5]]
# step 5 :re fit the model with the new x_optimal
sm_regressor = sm.OLS(exog = x_optimal , endog = y).fit()

# going back to step 3
# now let's see the p-value of each independant variable
sm_regressor.summary()

"""
variable   p-value

const      0.000
x1         0.940
x2         0.000
x3         0.604
x4         0.118
"""

# here we consider x1 and go to step 4
# step 4 we will remove x1 
# which is the secund dummy variable of state feature
# and its index is 1 from x_optimal
x_optimal = x[:,[0,3,4,5]]
# step 5 :re fit the model with the new x_optimal
sm_regressor = sm.OLS(exog = x_optimal , endog = y).fit()

# going back to step 3
# now let's see the p-value of each independant variable
sm_regressor.summary()

"""
variable   p-value

const      0.000
x1         0.000
x2         0.602
x3         0.105
"""

# step 3 here again x2 has the highest p-value 
# and it is > 0.05
 
# step 4 we will remove x2
# which is the administration feature
# and its index is 4 from x_optimal
x_optimal = x[:,[0,3,5]]
# step 5 :re fit the model with the new x_optimal
sm_regressor = sm.OLS(exog = x_optimal , endog = y).fit()

# going back to step 3
# now let's see the p-value of each independant variable
sm_regressor.summary()

"""
variable   p-value

const      0.000
x1         0.000
x2         0.060
"""

# step 3 here again x2 has the highest p-value 
# and it is > 0.05

# step 4 we will remove x2
# which is the 'marketing spend' feature
# and its index is 5 from x_optimal
x_optimal = x[:,[0,3]]
# step 5 :re fit the model with the new x_optimal
sm_regressor = sm.OLS(exog = x_optimal , endog = y).fit()

# going back to step 3
# now let's see the p-value of each independant variable
sm_regressor.summary()

"""
variable   p-value

const      0.000
x1         0.000
"""

# here all the variables have p-value < 0.05
# so this is the end and now The Model Is Ready
# so the conclusion is that 
# " R&D Spend feature has the highest statistical significant effect
# On the dependant variable profit"

# see all of the redundant lines above BAD THING isn't it ? 
# so let's implement the backward elimination algorithm method
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
    
# and we can use it like this
# select a significance level
sl = 0.05
# assign the output of the method to x_optimal
x_optimal = backwardElimination(x[:], sl)
# easy and nice ^^

# that's was Backward Elimination with considering p-values only
# Is there a better way ? well ,yes.

# Backward Elimination with p-values and Adjusted R Squared
def backwardEliminationWithAdjustedRSquared(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
    
# and we can use it like this
# select a significance level
sl = 0.05
# assign the output of the method to x_optimal
x_optimal = backwardEliminationWithAdjustedRSquared(x[:], sl)







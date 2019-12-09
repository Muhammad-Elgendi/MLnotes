# -*- coding: utf-8 -*-

"""
Preprocessing phase
what ?
why ?
how ?
"""

"""
libraries
"""
# pandas library is used to manage datasets
import pandas as pd
# matplotlib library is used to draw scatter and plot data points
import matplotlib.pyplot as plt
# numpy library is used to apply mathematics to our model
import numpy as np

"""
import dataset
"""
dataset = pd.read_csv('Data.csv')

"""
Spilt the features (independant variables) from dependant variable
"""
# assign independant variables to x
x = dataset.iloc[:,:-1].values
# assign dependant variable to y
y = dataset.iloc[:,-1].values

"""
Handle missing data techniques :
1- remove obeservation that has missing data
2- substitue the missing data with the mean of feature
3- substitue the missing data with the median of feature
4- substitue the missing data with the mode(most frequent value) of feature
"""

"""
Imputation :
In statistics, imputation is the process of replacing missing data with substituted values.
When substituting for a data point, it is known as "unit imputation"; when substituting for a component of a data point, it is known as "item imputation". 
"""
# import imuter class
from sklearn.preprocessing import Imputer
# create a new instance of imputer with the following parameters
imputer =Imputer(missing_values = 'NaN',strategy = 'mean', axis = 0)
# you can skip all these parameters since all of them are the default

# In python slicing the upperbound are excluded

# Fitting your model to (i.e. using the .fit() method on) the training data
# is essentially the training part of the modeling process.

# sometimes the "fit" method  is used for non-machine-learning methods, such as scalers and other preprocessing steps.
# In this case, you are merely "learn the required parameter" from your data

# The fit method modifies the object. And it returns a reference to the object. Thus, take care! 
"""
scikit-learn provides a library of transformers,
 Like other estimators, these are represented by classes with  a 
 
 fit method, which learns model parameters
 (e.g. mean and standard deviation for normalization)
 from a training set, and a 
 
 transform method which applies this transformation model to unseen data.

 fit_transform may be more convenient and efficient 
 for modelling and transforming the training data simultaneously.
"""
# calculate the mean for features that has missing values
imputer.fit(x[:,1:3])
# apply the imputation strategy to missing data
x[:,1:3] = imputer.transform(x[:,1:3])

"""
Encode the categorical and text data
"""
# Machine learning methods are based on some mathematical equations so
# we have to keep only numbers in our data
# so we have to encode text and categorical features

# Using label encoding
# import LabelEncoder
from sklearn.preprocessing import LabelEncoder
# create an instance of LabelEncoder class
labelEncoder = LabelEncoder()

"""
used to transform non-numerical labels 
(as long as they are hashable and comparable) to numerical labels.
"""

# apply lablel encoding to categorical data
x[:,0] = labelEncoder.fit_transform(x[:,0])

# Note ! : since country feature catagories has no relation
# between each other and can't be comparable 
# or put in any kind of order (e.g. sizes small,medium,large) ,
# so we can't apply this type of encoding

# we will use other type of Encoding called Dummy or OneHot Encoding
# Dummy variables trap ?

# Using OneHot encoding aka one-of-K scheme
# import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# create an instance of OneHotEncoder class with categorical_features
# parameters is an array that has the indexs of categorical data
oneHotEncoder = OneHotEncoder(categorical_features = [0])

"""
used to transform non-numerical labels 
(as long as they are hashable and comparable) to numerical labels.
"""

# apply OneHot encoding to categorical data (country feature)
x = oneHotEncoder.fit_transform(x).toarray()
# apply label encoding to the dependant variable y
y = labelEncoder.fit_transform(y)

"""
Splitting the Dataset into the Training set and Test set
why ?
We try to figure out the correlations among features from training set and
build a model that will learn these correlations and then
we test its preformance with the new test set which has a slightly 
different correlations to see how the model accuracy is for new data
"""

# import train_test_split class
from sklearn.cross_validation import train_test_split
# train_test_split returns List containing train-test split of inputs.
# In python you can assign muliple variable to a list
# so m,n = [0,1] means that m = 0 and n = 1
# we can also set random_state parameter to a number so every time 
# we run the script sampling will be the same
# good test size is from 0.2 to 0.4 (in some rare cases)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

"""
Feature Scaling
Why ?

Let's just focus on the age and the salary features.
You notice that the variables are not on the same scale because
the age are going from 27 to 50.
And the salaries going from 40 K to like 90 K..

So because this age variable in the salary variable don't have 
the same scale.
This will cause some issues in your models.

And why is that ?

It's because a lot of models are based on what is called 
the Euclidean distance.
for other algorithms that are't based on the Euclidean distance
feature scaling make them run faster.

How we put all the feature in the same scale ?
1- standardisation
2- normalization

featue scaling may be applied to 
1- Dummy variables (Optional)
2- dependant variable in regression problems

Note ! some libraries apply feature scaling for you
"""
# import StandardScaler class
from sklearn.preprocessing import StandardScaler
# create a StandardScaler instance called scaler
scaler = StandardScaler()
# calculate scale and apply it to x_train
x_train = scaler.fit_transform(x_train)
# apply scale to x_test 
# (No need to call fit() since scale is already calculated)
x_test = scaler.transform(x_test)


























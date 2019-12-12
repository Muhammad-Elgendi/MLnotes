#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Regression
"""
# import dataset
# import pandas library and give it "pd" as alias
import pandas as pd
# read dataset from csv file
dataset = pd.read_csv("Social_Network_Ads.csv")

# show first five observation of the dataset
# print(dataset.head())
# To see the statistical details of the dataset, we can use describe():
# print(dataset.describe())

# we are intersted in Age ,EstimatedSalary ,and Purchased features
# split the dataset to dependant and independant variables
# dependant variable is Purchased
# y = dataset.iloc[:,-1].astype(float).values
# here we used astype() which Cast a pandas object to
# a specified dtype. and we want to cast it float so
# that we can apply feature scaling with StandardScaler class
# without any warning
y = dataset.iloc[:,-1]
# independant variables are Age ,EstimatedSalary
x = dataset.iloc[:,2:-1]

# split dataset to training set and test set
# import train_test_split class
from sklearn.cross_validation import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,
                                                 random_state = 42)

# apply feature scaling to independant variables x
# import StandardScaler class
from sklearn.preprocessing import StandardScaler
# create instance of StandardScaler
xScaler = StandardScaler()
# scale x_train
x_train = xScaler.fit_transform(x_train)
# scale x_test to the same scale that fitted to x_train
x_test = xScaler.transform(x_test)

# create a LogisticRegression classifier
# import LogisticRegression class
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)

# fit (train) the classifier into training set
classifier.fit(x_train,y_train)

# test the classifier on the test set
y_predicted = classifier.predict(x_test)

# calculate the confusion matrix
"""
What is confusion matrix ?

"""
# import confusion_matrix class
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(y_test,y_predicted)
# print confusion_matrix
print(confusionMatrix)

# split observations to purchased and not purchased
# filter out the purchased observations
purchased = dataset.loc[y == 1]
# filter out the not purchased observations
not_purchased = dataset.loc[y == 0]

# visualize the dataset observations only
# import matplotlib.pyplot and give it "plt" as an alias
import matplotlib.pyplot as plt
# create new figure
plt.figure()
# plot purchased observations in green color
plt.scatter(purchased.iloc[:,3], purchased.iloc[:,2], label='Purchased' ,color="blue")
# plot not_purchased observations in red color
plt.scatter(not_purchased.iloc[:,3], not_purchased.iloc[:,2], label='Not Purchased' ,color="brown")
plt.title('Age vs Estimated Salary')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
# adjust the padding of the figure
plt.tight_layout()
# export the figure to an image
plt.savefig("Age vs Estimated Salary.png")

# visualize the model with the test set
# import numpy and give it an alias "np"
import numpy as np
from matplotlib.colors import ListedColormap

# create new figure
plt.figure()

# assign x and y sets to your sets
# (Note! these sets must be a numpy array)
X_set, y_set = x_test, y_test.values

# create coordinate matrices from coordinate vectors.(mesh grid)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# plot the model decision boundary
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

# plot scaled observations
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
# set limits of axes
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# show the legend of the figure
plt.legend()

# set title and axes labels
plt.title('Age vs Estimated Salary (Test set classification)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

# adjust the padding of the figure
plt.tight_layout()

# export the figure to an image
plt.savefig("Age vs Estimated Salary (Test set classification).png")

# show the figure
plt.show()


# visualize the model with the training set

# create new figure
plt.figure()

# assign x and y sets to your sets
# (Note! these sets must be a numpy array)
X_set, y_set = x_train, y_train.values

# create coordinate matrices from coordinate vectors.(mesh grid)
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# plot the model decision boundary
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

# plot the model decision boundary (Another way)
# compute points on the line by appling the equation of the line)
# get the coefficient of the features 
w = classifier.coef_[0]
a = -w[0] / w[1]
# set x coordinate of the points
xx = np.linspace(-5, 5)
# linspace() Return evenly spaced numbers over a specified interval.
# compute y coordinate of the points
yy = a * xx - (classifier.intercept_[0]) / w[1]
# plot the line in black color
plt.plot(xx, yy, color="black")

# plot scaled observations
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
# set limits of axes
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# show the legend of the figure
plt.legend()

# set title and axes labels
plt.title('Age vs Estimated Salary (Training set classification)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

# adjust the padding of the figure
plt.tight_layout()

# export the figure to an image
plt.savefig("Age vs Estimated Salary (Training set classification).png")

# show the figure
plt.show()


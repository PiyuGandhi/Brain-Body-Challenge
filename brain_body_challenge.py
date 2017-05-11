import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv('brain_body_challenge.txt',names=['Brain','Body'])
data_test = pd.read_fwf('brain_body.txt')

# Training Features and Labels
X_Train = data_train[['Brain']]
Y_Train = data_train[['Body']]

# Test Features and Labels
X_Test = data_test[['Brain']]
Y_Test = data_test[['Body']]

# Linear Regression

from sklearn.linear_model import LinearRegression

# Fitting the data
reg = LinearRegression()
reg.fit(X_Train,Y_Train)

# Predicting
pred = reg.predict(X_Test)

# Visualization
# Training Data
plt.scatter(X_Train,Y_Train)

plt.plot(X_Test,reg.predict(X_Test),'b')
plt.show()

print "Accuracy Linear :- " , reg.score(X_Test,Y_Test)


# Lasso Regression

from sklearn.linear_model import Lasso

# Fitting the data
reg = Lasso()
reg.fit(X_Train,Y_Train)

# Predicting
pred = reg.predict(X_Test)

# Visualization
# Training Data
plt.scatter(X_Train,Y_Train)

plt.plot(X_Test,reg.predict(X_Test),'b')
plt.show()

print "Accuracy Lasso :- " , reg.score(X_Test,Y_Test)
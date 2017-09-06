# -*- coding: utf-8 -*-
"""
#   -------------------------------------------------   #
#
#   -------- Description --------
#   Example of LinearRegression with p = 5
#   p: Anzahl der Variablen
#
#   -------- Run --------
#
#
#
#   Created on Mon Sep  4 14:06:29 2017 by   mom
#   -------------------------------------------------   #
"""

import numpy as np
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pylab as plt

# --- Parameter
X_limit = {'p0': [0., 10.],
           'p1': [0., 10.],
           'p2': [0., 10.],
           'p3': [0., 10.],
           'p4': [0., 10.]}

# --- generate DOE-tets
def test_process(X_limit,n):
    """
    n - Anzahl der Versuche z.B. n = 100
    """
    X = []
    for key, limit in X_limit.items():
        X.append(np.random.uniform(limit[0],limit[1], n))

    return np.array(X).T


# --- System definition
def System(X_i):
    """
    Input X_i = [0.4, 1, ... ,10]
    """
    assert X_i.shape[1] == 5, "Fehler, X_i hat nicht die Richtige anzahl\
               an Elementen"
    return 5*X_i[:,0]**2 * X_i[:,1] - X_i[:,1]*0.5 + 3*X_i[:,2]**2 \
               + 1 * X_i[:,3] + 10* X_i[:,4]

# -----------------------------------------------------------
#                   Regression
# -----------------------------------------------------------


# --- run simulation
X = test_process(X_limit,100)
y = System(X)

# --- split in train and test
X_train, X_test, y_train, y_test = train_test_split(\
                    X, y, test_size=0.25, random_state=42)

# --- normalize
#X_norm =  StandardScaler(X)

# --- linear regression
regr = LinearRegression()
regr.fit(X_train,y_train)

beta_ = regr.coef_
y_ = regr.predict(X_test)



# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f\n\n' % r2_score(y_test, y_))



plt.scatter(X_test[:,1], y_test)
plt.scatter(X_test[:,1],y_)
plt.legend([r'$y_{test}$', r'$\hat{y}$'])
plt.show()


# -----------------------------------------------------------
#                   Regression
# -----------------------------------------------------------
print('Ergebnis nicht zufrieden stellend\n\n')

# --- transfrom X1, X2, ... zu X1,X2... X1**2, X2**2,... X1X2, ...
X_train_poly = PolynomialFeatures(degree=2)
X_train_poly = X_train_poly.fit_transform(X)


# --- split in train and test
X_train, X_test, y_train, y_test = train_test_split(\
                    X_train_poly, y, test_size=0.25, random_state=42)

# --- normalize
#X_norm =  StandardScaler(X)

# --- linear regression
regr = LinearRegression()
regr.fit(X_train,y_train)

beta_ = regr.coef_
y_ = regr.predict(X_test)



# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f\n\n' % r2_score(y_test, y_))



plt.scatter(X_test[:,1], y_test)
plt.scatter(X_test[:,1],y_)
plt.legend([r'$y_{test}$', r'$\hat{y}$'])
plt.show()
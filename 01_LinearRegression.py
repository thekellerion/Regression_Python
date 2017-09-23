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
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# --- Parameter
X_limit = {'a': [-10., 10.],
           'b': [-10., 10.],
           'c': [-10., 10.],
           'd': [0., 10.],
           'e': [0., 10.]}

# --- generate DOE-tets
def test_process(X_limit,n):
    """
    n - Anzahl der Versuche z.B. n = 100
    Output: Dataframe
    """
    # --- run for each key item in X_limit a random uniform distr. n-times
    X = []
    column = []
    for key, limit in X_limit.items():
        X.append(np.random.uniform(limit[0],limit[1], n))
        column.append(key)
    # --- create Pandas Dataframe with column name from X_limit
    X = pd.DataFrame(np.array(X).T)
    X.columns = column

    return X


# --- System definition
def System(X):
    """
    Input X
    """
    return 5*X['a']**2 * X['b'] - X['b']*0.5 + 3*X['c']**2 \
               + 1 * X['d'] + 10* X['e']

# -----------------------------------------------------------
#                   Regression
# -----------------------------------------------------------
def RegressionResult(X, y):
    """
    X: Input
    y: response

    1) split in Train and Test
    2) run LinearRegression, without intercept!!
    3) run prediction
    4) print Mean Sqared error and r2-score
    5) plot X[1] and y;   plot beta

    Y_ = beta_0 + beta_1 *x_1 ...+ beta_k * x_k
    Y_ = beta_*X
    wobei --> beta_0 = 0
    return beta_ (coef_)
    """
    # --- split in train and test
    X_train, X_test, y_train, y_test = train_test_split(\
                        X, y, test_size=0.25, random_state=42)

    # --- linear regression
    regr = LinearRegression(fit_intercept=False)
    regr.fit(X_train,y_train)

    beta_ = regr.coef_
    y_ = regr.predict(X_test)

    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_))

    # Explained variance score: 1 is perfect prediction
    print('R^2 score coefficient of determination: %.2f\n\n' % r2_score(y_test, y_))

    plt.scatter(X_test['a'], y_test)
    plt.scatter(X_test['a'],y_)
    plt.legend([r'$y_{test}$', r'$\hat{y}$'])
    plt.xlabel('b')
    plt.show()

    ind = np.arange(len(beta_))  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(ind, beta_, width)
    ax.set_xticklabels(X.columns)
    ax.set_xticks(ind + width / 2)
    plt.grid()
    plt.show()

    beta_df = pd.DataFrame([X.columns, beta_]).T
    beta_df.columns = ['beta', 'value']
    beta_df = beta_df.set_index('beta')
    return beta_df


def show_important_coef(Beta, n=10):
    """
    shows the n imporant coefficient of the regression
    Beta = DataFrame
    [beta, value]
    bsp:   Beta.loc[['d']]  ->    10
    """
    t = Beta.sort_values('value', ascending = False)
    t = t[0:n]

    t.plot(kind='bar')





# --- run simulation
X = test_process(X_limit,5)
y = System(X)

RegressionResult(X, y)

print('-'*80)
print('Ergebnis nicht zufrieden stellend.')
print('Besser polynom ansatz wählen: 1,X2... X1**2, X2**2,... X1X2, ...')
print('-'*80)

# --- transfrom X1, X2, ... zu X1,X2... X1**2, X2**2,... X1X2, ...
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_poly = pd.DataFrame(X_poly)
X_poly.columns = poly.get_feature_names(X.columns)    # keep names

beta = RegressionResult(X_poly, y)


print ('die Wichtigsten Beta')
show_important_coef(beta)
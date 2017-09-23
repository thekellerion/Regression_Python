# -*- coding: utf-8 -*-
"""
#   -------------------------------------------------   #
#
#   -------- Description --------
#   Lasso - identify not requiered parameter
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
from sklearn.linear_model import Lars, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pylab as plt
import seaborn as sns

# --- Parameter
X_limit = {'a': [-5., 10.],
           'b': [0., 10.],
           'c': [-5., 10.],
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
               + 1 * X['d']**-1 + 10* X['e'] + 100


# -----------------------------------------------------------
#                   Regression
# -----------------------------------------------------------
def RegressionResult(runs = 100, kind='Linear'):
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

    sim_list = pd.DataFrame()

    for i in range(runs):
            # --- run simulation
        X = test_process(X_limit,100)
        y = System(X)

        # --- transfrom X1, X2, ... zu X1,X2... X1**2, X2**2,... X1X2, ...
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        X_poly = pd.DataFrame(X_poly)
        X_poly.columns = poly.get_feature_names(X.columns)    # keep names


        # --- split in train and test
        X_train, X_test, y_train, y_test = train_test_split(\
                            X_poly, y, test_size=0.25, random_state=42)

        if kind == 'Linear':
            # --- linear regression
            regr = LinearRegression(fit_intercept=False)
        elif kind == 'Lars':
            regr = Lars(fit_intercept=False, n_nonzero_coefs=7)
        else:
            raise Exception('Error: Falscher Regressionsart gewählt')

        regr.fit(X_train,y_train)

        beta_ = regr.coef_
        y_ = regr.predict(X_test)

        act_sim = pd.DataFrame([beta_] ,index = [i])
        act_sim.columns = X_poly.columns
        act_sim['R2_Score'] = r2_score(y_test, y_)

        sim_list = pd.concat([sim_list,act_sim])


    return sim_list

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

runLinear = RegressionResult()
runLars = RegressionResult(kind='Lars')

sns.set_style("whitegrid")

# --- Plot Variation von Beta über die n-Simulationen
ax = sns.boxplot(data=runLinear)
ax.set_title('Coeff runLinear - 100 Runs')
ax.set(ylim=(-300, 300))
plt.show()
ax = sns.boxplot(data=runLars)
ax.set_title('Coeff runLasso - 100 Runs')
ax.set(ylim=(-50, 30))
plt.show()

# --- plot R2 vergleich zeischen Linear und Lasso
temp = pd.DataFrame()
temp['R2_LinearRegression'] = runLinear['R2_Score']
temp['R2_LassoRegression'] = runLars['R2_Score']

ax = sns.boxplot(data=temp)

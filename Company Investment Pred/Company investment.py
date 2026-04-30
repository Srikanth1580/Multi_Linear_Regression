import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(script_dir, "Investment.csv"))

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

x= pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
x_train , x_test, y_train,_y_test = train_test_split(x,y, train_size=0.8, random_state=1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

m= regressor.coef_
print(m)
c=regressor.intercept_
print(c)

# x= np.append(arr= np.ones((50,1)).astype(int), values=x, axis= 1)
x= np.append(arr= np.full((50,1),49834).astype(int), values=x, axis= 1)

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1,3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
x_opt = x[:,[0,1]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

filename = 'linear_regression_model_companyinvestment_pred.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as inear_regression_model_companyinvestment_pred.pkl")

import os
os.getcwd()


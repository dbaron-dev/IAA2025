#Importing numpy and scipy
import numpy as np
from scipy.stats import t

#Regression class

class LinearRegressionEx:
   def __init__(self):
       self.coefficients = None
       self.p_values = None


   def fit(self, X, y):
       # Add a column of ones to X for the intercept term
       one = np.ones((len(X))).reshape(len(X),1)
       X = np.hstack((one,X))


       # Calculate the coefficients using the normal equation
       self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y


   def p_values_cal(self, X, y):
       one = np.ones((len(X))).reshape(len(X),1)
       X = np.hstack((one,X))
       # Calculate the residuals
       y_pred = X @ self.coefficients
       residuals = y - y_pred


       # Calculate the residual sum of squares (RSS)
       RSS = np.sum(residuals ** 2)


       # Calculate the degrees of freedom
       n = X.shape[0]
       p = X.shape[1] - 1
       df = n - p - 1


       # Calculate the standard error of the coefficients
       XTX_inv = np.linalg.inv(X.T @ X)
       coef_se = np.sqrt(np.diagonal(XTX_inv) * RSS / df)
       # Calculate the t-statistic and p-value for each coefficient
       t_stat = self.coefficients / coef_se
       p_values = (1 - t.cdf(np.abs(t_stat), df)) * 2


       self.p_values = p_values
       return p_values


   def predict(self, X):
       # Add a column of ones to X for the intercept term
       one = np.ones((len(X))).reshape(len(X),1)


       X = np.hstack((one,X))


       # Calculate the predicted values
       y_pred = X @ self.coefficients


       return y_pred


# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:04:47 2016

@author: jarektb
"""
import pandas as pd
class Supervised_Models:
    def __init__(self, pandas_object):
        self.pandas_data = pandas_object
    def getLinearRegressionModel():        
        %matplotlib inline
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn; 
        from sklearn.linear_model import LinearRegression
        import pylab as pl
        
        seaborn.set()
        import numpy as np
        np.random.seed(0)
        X = np.random.random(size=(20, 1))
        y = 3 * X.squeeze() + 2 + np.random.randn(20)
        
        plt.plot(X.squeeze(), y, 'o');
        model = LinearRegression()
        model.fit(X, y)
        
        # Plot the data and the model prediction
        X_fit = np.linspace(0, 1, 100)[:, np.newaxis]
        y_fit = model.predict(X_fit)
        
        plt.plot(X.squeeze(), y, 'o')
        plt.plot(X_fit.squeeze(), y_fit);
        
        
        
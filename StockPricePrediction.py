# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:02:40 2019

@author: nikhil
"""

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = 'Cje7mXxsjgxUnVxTtmCN'

df = quandl.get('EOD/NKE')

df = df[['Adj_Open', 'Adj_High','Adj_Low','Adj_Close','Adj_Volume']]
df['HL_PCT']=(df['Adj_High'] - df['Adj_Low'])*100/df['Adj_Close']
df['PCT_CHANGE'] = (df['Adj_High'] - df['Adj_Low'])*100/df['Adj_Low']

df = df[['Adj_Close','HL_PCT','PCT_CHANGE', 'Adj_Volume']]

fcastCol = 'Adj_Close'
df.fillna(-9999, inplace=True)

forecast = int(math.ceil(0.01*len(df)))
df['label'] = df[fcastCol].shift(-forecast)


X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast]
X_latest = X[-forecast:]
df.dropna(inplace=True)

#Normalizing input data, takes time, can avoid if dealing with HFT
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
acc = clf.score(X_test,y_test)

fcast = clf.predict(X_latest)
df['Prediction'] = np.nan

endDate = df.iloc[-1].name
end_unix = endDate.timestamp()
nextUnix = end_unix + 86400

for i in fcast:
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += 86400
    df.loc[nextDate] = [np.nan for _ in range(len(df.columns)-1)] + [i]
df['Adj_Close'].plot()
df['Prediction'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



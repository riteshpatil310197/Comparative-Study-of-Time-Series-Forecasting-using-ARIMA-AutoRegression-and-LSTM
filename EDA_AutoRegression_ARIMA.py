# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:52:01 2022

@author: Ritesh
"""
'''RESULTS:
    RMSE for AutoRegression:6.4446 for lags=3
    '''

# Import and Read Data
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow.keras import optimizers

df=pd.read_csv("C:\\Users\\Ritesh\\MSc Data Analytics\\Thesis\\observations.csv",index_col=6,parse_dates=True)

df.head()
new_df=df.loc[df.station=="ATHENRY",]

#train_dates= pd.to_datetime(new_df['date'])

#type(new_df['date'])
new_df=new_df.loc[:,['temp','wdsp','wddir']]
new_df=new_df.dropna(axis=0)


#Auto Regression
from statsmodels.tsa.ar_model import AutoReg
new_df_AR=new_df.loc[:,['temp']]
new_df_AR.index = pd.DatetimeIndex(new_df_AR.index).to_period('H')


#Plot Temperatures
new_df_AR.plot()

#Check for Stationarity-ADF test

from statsmodels.tsa.stattools import adfuller
adf_AR=adfuller(new_df_AR['temp'])

print("P-value:"+str(adf_AR[1])) 
# Reject Null Hypothesis: That is, Given TS is stationary

#Plot Partial Correlation

from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
#When using AR model, only consider Partial Auto correlation plot.
'''Partial Auto correlation considers direct impact of n'th lag on current prediction, while
Auto correlation considers all the indirect effects till n'th lag on current prediction'''

pacf_AR=plot_pacf(new_df_AR['temp'], lags=25)
#Lag till 4 looks reasonable
acf_AR=plot_acf(new_df_AR['temp'], lags=25)

#Train Test Split
train_AR=new_df_AR[:len(new_df_AR)-16]
test_AR=new_df_AR[len(new_df_AR)-16:]

#Build Auto Regression Model
#Experiment with lags for better rmse
model_AR=AutoReg(train_AR, lags=3).fit()
model_AR.summary()
#From summary, check p values for 4 lags. If all of them are less than 0.05, it indicates all lags were important.

#Predictions for AR
pred_AR=model_AR.predict(start=len(train_AR),end=len(train_AR)+len(test_AR)-1,dynamic=False)

'''plt.plot(pred_AR.values, color='red')
plt.plot(test_AR.values, color='green')
#plt.legend()
plt.show()'''

#Calculate Error for AR
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse=sqrt(mean_squared_error(test_AR, pred_AR))






#ARIMA Model
from pmdarima import auto_arima
arima_fit=auto_arima(new_df_AR['temp'],
                     trace=True,suppress_warnings=True)
arima_fit.summary()

#Fit and Predict-ARIMA
from statsmodels.tsa.arima_model import ARIMA
model_ARIMA=ARIMA(train_AR, order=(3,1,2))
model_ARIMA=model_ARIMA.fit()
model_ARIMA.summary()

#Predict ARIMA
start=len(train_AR)
end=len(train_AR)+len(test_AR)-1
pred_ARIMA=model_ARIMA.predict(start=start,end=end,typ="levels")

rmse_ARIMA=sqrt(mean_squared_error(test_AR, pred_ARIMA))
plt.plot(pred_AR.values, color='red')
plt.plot(pred_ARIMA.values, color='blue')
#plt.plot(test_AR.values, color='green')
#plt.legend()
plt.show()
test_AR.index=pred_AR.index


ri=pd.concat([test_AR,pred_AR,pred_ARIMA],axis=1)

ri.rename(columns={'temp':'Actual',0:'AR Prediction',1:'ARIMA Prediction'},inplace=True)

ri.plot()
plt.legend(loc=(1.04,0.5))
plt.show()

#ETS





















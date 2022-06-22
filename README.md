# Description
This Project focusses on comparing performance of **ARIMA, Autoregression and LSTM** to forecast temperature of Ireland. 

## DataSet
The Dataset contains hourly data of Ireland for the year 2017. Dataset can be found at: https://CRAN.R-project.org/package=aimsir17

## Overview of AutoRegression, ARIMA and LSTM

### AutoRegression:
Auto Regressive (AR) model is a model used in time series forecasting where we forecast the variable of Interest using a linear combination of past values of the variable. The term autoregression indicates that it is a regression of the variable against itself.

### ARIMA Model:
An ARIMA model can be understood by outlining each of its components as follows:

Autoregression (AR): refers to a model that shows a changing variable that regresses on its own lagged, or prior, values.

Integrated (I): represents the differencing of raw observations to allow for the time series to become stationary (i.e., data values are replaced by the difference between the data values and the previous values).

Moving average (MA):  incorporates the dependency between an observation and a residual error from a moving average model applied to lagged observations.
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:13:49 2022

@author: Ritesh
"""

import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM,Dropout
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow.keras import optimizers

df=pd.read_csv("C:\\Users\\Ritesh\\MSc Data Analytics\\Thesis\\observations.csv",parse_dates=True,index_col='date')

df.head()
new_df=df.loc[df.station=="ATHENRY",]

#train_dates= pd.to_datetime(new_df['date'])

#type(new_df['date'])
#new_df=new_df.loc[:,['date','temp','wdsp','wddir']]

#new_df.plot.line()
#new_df['temp'].plot.line()
new_df_train=new_df.loc[:,['temp']]

new_df_train=new_df_train.dropna(axis=0)

train=new_df_train[:-16]
test=new_df_train[-16:]

scaler=StandardScaler()
scaler=scaler.fit(train)
train=scaler.transform(train)
test=scaler.transform(test)

#train=new_df_train_scaled[:-16]
#test=new_df_train_scaled[-16:]

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
n_input=15
n_features=1
generator=TimeseriesGenerator(train, train, length=n_input,batch_size=1)

#Verify generator
X,y=generator[0]
print(X.flatten())
print(y)

#Model
'''model=Sequential()
model.add(LSTM(64,activation='relu',input_shape=(n_input,n_features),return_sequences=True))
model.add(LSTM(32,activation='relu',return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.summary()

model.fit(generator,epochs=10)

last_train_batch=train[-15:]
last_train_batch=last_train_batch.reshape((1,n_input,n_features))
model.predict(last_train_batch)

test[0]'''
model=load_model('lstm_univariate.h5')

test_predictions=[]
first_eval_batch=train[-15:]
current_batch=first_eval_batch.reshape((1,n_input,n_features))

for i in range(len(test)):
    current_pred=model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch=np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
    
prediction_trans=scaler.inverse_transform(test_predictions)
actual_test=scaler.inverse_transform(test)
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse=sqrt(mean_squared_error(actual_test, prediction_trans))

'''plt.plot(actual_test,label="Original Data")
plt.plot(prediction_trans,label="Predicted by LSTM")
plt.legend()
plt.show()'''
actual_test=actual_test.reshape(16,)
prediction_trans=prediction_trans.reshape(16,)
rilu=pd.concat([pd.Series(actual_test),pd.Series(prediction_trans)],axis=1)
rilu.index=df.index[-16:]
rilu.columns=['Actual','LSTM Univariate Predictions']
rilu.plot()
model.save('lstm_univariate.h5') # SAVED IN COMPUTER VISION FOLDER




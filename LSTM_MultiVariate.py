import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow.keras import optimizers

df=pd.read_csv("C:\\Users\\Ritesh\\MSc Data Analytics\\Thesis\\observations.csv")

df.head()
new_df=df.loc[df.station=="ATHENRY",]

train_dates= pd.to_datetime(new_df['date'])

#type(new_df['date'])
new_df=new_df.loc[:,['date','temp','wdsp','wddir']]

#new_df.plot.line()
#new_df['temp'].plot.line()
new_df_train=new_df.loc[:,['temp','wdsp','wddir']]

new_df_train=new_df_train.dropna(axis=0)

scaler=StandardScaler()
scaler=scaler.fit(new_df_train)
new_df_train_scaled=scaler.transform(new_df_train)

#create separate time series

n_future=1
n_past=14

train_X=[]
train_Y=[]

for i in range(n_past,len(new_df_train_scaled)-n_future+1):
    train_X.append(new_df_train_scaled[i-n_past:i,0:new_df_train_scaled.shape[1]])
    train_Y.append(new_df_train_scaled[i+n_future-1:i+n_future,0])
    
train_X,train_Y=np.array(train_X),np.array(train_Y)

test_X=train_X[-16:,:,:]
test_Y=train_Y[-16:,]

train_X=train_X[:-16,:,:]
train_Y=train_Y[:-16,]


#Create sequential model(need to adjust hyperparameters)
model=Sequential()
model.add(LSTM(64,activation="relu",input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences=True))
model.add(LSTM(32,activation="relu",return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(train_Y.shape[1]))

#optimizer = optimizers.Adam(clipvalue=0.5,learning_rate=0.0001)
model.compile(optimizer='adam',loss='mse')
model.summary()

history = model.fit(train_X, train_Y, epochs=10,validation_split=0.1, verbose=1)


'''plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()'''

forecast_data=train_X[-16:]

prediction=model.predict(test_X)

prediction_copies = np.repeat(prediction, new_df_train_scaled.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]

#a=train_Y[-16:]
y_test=np.repeat(test_Y, new_df_train_scaled.shape[1], axis=-1)
y_test=scaler.inverse_transform(y_test)[:,0]
'''plt.plot(y_test,label="Original Data")
plt.plot(y_pred_future,label="Predicted by LSTM")
plt.legend()
plt.show()'''

rilm=pd.concat([pd.Series(y_test),pd.Series(y_pred_future)],axis=1)
rilm.index=df['date'][-16:]

rilm.columns=['Actual','LSTM Multivariate Prediction']
rilm.plot()





from math import sqrt
from sklearn.metrics import mean_squared_error
rmse=sqrt(mean_squared_error(y_test, y_pred_future))
    
    











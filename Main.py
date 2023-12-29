# github : CodeYard01

#importing libraires
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web
import datetime as dt 

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#Stock Or CryptoCurrency
stock = 'FB'

start = dt.datetime(2020,1,1)
end = dt.datetime(2023,12,29)

data = web.DataReader(stock,'yahoo',start,end)

#prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#model dev

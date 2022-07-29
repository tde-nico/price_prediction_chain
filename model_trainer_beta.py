import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


CSV = 'BTC-USD.csv'
TRAIN_SPLIT = 0.8
PREDICTION_DAYS = 60
MODEL = 'test.h5'


df = pd.read_csv(CSV)

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * TRAIN_SPLIT)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(PREDICTION_DAYS, len(train_data)):
	x_train.append(train_data[i-60:i, 0])
	y_train.append(train_data[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

### Neural Network ###
'''
neurons = 50
model = Sequential()
model.add(LSTM(neurons, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(neurons, return_sequences=False))
model.add(Dense(neurons / 2))
model.add(Dense(1))
'''
### Neural Network ###

print(len(x_train), x_train.shape[1])

model_2 = Sequential()
model_2.add(LSTM(
    units=50,
    return_sequences=True,
    input_shape=(x_train.shape[1], 1)
))
model_2.add(Dropout(0.2)) # prevent overfitting
model_2.add(LSTM(
    units=50,
    return_sequences=True
))
model_2.add(Dropout(0.2)) # prevent overfitting
model_2.add(LSTM(
    units=50
))
model_2.add(Dropout(0.2)) # prevent overfitting
model_2.add(Dense(units=1)) # final value

### Neural Network ###


#model.compile(optimizer='adam', loss='mean_squared_error')
#model.fit(x_train, y_train, batch_size=1, epochs=20)

#model.save(MODEL)


model_2.compile(optimizer='adam', loss='mean_squared_error')
model_2.fit(x_train, y_train, epochs=25, batch_size=32)

model_2.save("2_" + MODEL)


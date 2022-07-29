import math
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
import keras

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


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


models = [keras.models.load_model(MODEL), keras.models.load_model("2_" + MODEL)]



new_df = data
last_60_days = new_df[-60:].values

last_60_days_scaled = scaler.transform(last_60_days)

x_test = []
x_test.append(last_60_days_scaled)
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

for i in range(60):

  #print(x_test)
  #print(last_60_days_scaled)

  pred_price = models[i % 2].predict(x_test)
  
  #print()
  pred_price = pred_price[0][0]
  x_test[0] = np.reshape(np.append(np.delete(x_test[0], 0), [pred_price]), (x_test[0].shape[0], 1))
  #print(x_test)
  #print()
  #x_test[0] = np.append(x_test[0], pred_price[0])
  #print(x_test)
  #print()
  #last_60_days_scaled = np.delete(last_60_days_scaled, 0)
  #last_60_days_scaled = np.append(last_60_days_scaled, pred_price[0])
  #last_60_days_scaled = np.reshape(last_60_days_scaled, (last_60_days_scaled.shape[0], 1))
  #break

#last_60_days_scaled = np.reshape(last_60_days_scaled, (last_60_days_scaled.shape[0], 1))
x_test = np.reshape(x_test, (x_test.shape[1], 1))
pred_price = scaler.inverse_transform(x_test)
#print(pred_price, pred_price.shape)


pred = pd.DataFrame(pred_price, columns=['Close'])
pred['Date'] = pd.date_range(dt.datetime.now().date(), periods=60, freq='D')
pred = pred.set_index(pred['Date'])

print(pred)
print(df)
print(df['Date'])

plt.figure(figsize=(20,10))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('USD [$]', fontsize=18)
#plt.plot(df['Date'], df['Close'])
plt.plot(pred['Close'])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

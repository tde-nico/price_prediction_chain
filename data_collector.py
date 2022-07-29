import pandas_datareader as web
import datetime as dt
import pandas as pd

currency = 'BTC-USD'
start = '2019-01-01'

df = web.DataReader(
    currency,
    data_source='yahoo',
    start=start,
    end=dt.datetime.now()
)

df.to_csv(currency + '.csv')

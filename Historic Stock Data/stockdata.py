import pandas as pd
from alpha_vantage.alpha_vantage.timeseries import TimeSeries
import time

api_key = 'API KEY'

ts = TimeSeries(key= api_key, output_format='pandas')
data, meta_data = ts.get_daily_adjusted(symbol = 'SYMBOL', outputsize = 'full')
print(data)

i = 1
while i == 1:
    data, meta_data = ts.get_daily_adjusted(symbol = 'SYMBOL',  outputsize = 'full')
    print(data).to_excel("output.xlsx")
    time.sleep(60)

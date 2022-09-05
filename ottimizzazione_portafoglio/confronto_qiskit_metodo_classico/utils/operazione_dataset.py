import numpy as np
import pandas as pd
import yfinance as yf

payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
second_table = payload[1]

df = first_table
symbols = df['Symbol'].values.tolist()
print(len(symbols))
test = yf.download(symbols, period='1y')
test.head()
test_2 = test[test.columns[1:8]]
test_2 = test_2['Adj Close']
print(test_2)
test_2.to_csv('dataset_piu_variabili.csv')
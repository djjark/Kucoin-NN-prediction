# import pandas as pd
# from collections import deque
# from datetime import datetime
# import random
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# import time
# from sklearn import preprocessing
# from tensorflow.keras.models import Sequential
# import os

# SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
# FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
# RATIO_TO_PREDICT = "ETH-USD"
# EPOCHS = 10
# BATCH_SIZE = 16
# # ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
# ratios = ["BTC-USD"]  # the 4 ratios we want to consider

# main_df = pd.DataFrame() # begin empty


# for ratio in ratios:  # begin iteration
#     dataset = f'crypto_data/{ratio}.csv'  # get the full path to the file.
#     df = pd.read_csv(dataset, names=)



import json
import pandas as pd
import ccxt
from datetime import datetime
with open('D:/TensorFlow/GitHub projects/Kucoin/codes.json') as f:
	data = json.load(f)
	

hitbtc   = ccxt.hitbtc({'verbose': True})
# bitmex   = ccxt.bitmex()
# huobipro = ccxt.huobipro()

exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
		'apiKey': data['api_key'],
		'secret': data['secret_api_pass'],
		'timeout': 30000,
		'enableRateLimit': True,
})

def date(var):
    return datetime.utcfromtimestamp(int(var)).strftime('%Y-%m-%d %H(h)%M(m)%S(s)')

# hitbtc_markets = hitbtc.load_markets()
hour = []
coin = 'ALGO'
pair = 'USDT'
for i in exchange.load_markets():
	if i == f'{coin}/{pair}':
		hour = exchange.fetchOHLCV(i,timeframe='1h')
		break

df = pd.DataFrame(hour, columns=['Time','Open','High','Low','Close','Volume'])
df.to_csv(f'D:\TensorFlow\GitHub projects\Kucoin\Kucoin NN prediction\crypto_data\{coin}-{pair}.csv', index=False, header=False)
# print(df)
# print(huobipro.id, huobipro.load_markets())

# print(hitbtc.fetch_order_book(hitbtc.symbols[0]))
# print(bitmex.fetch_ticker('BTC/USD'))
# print(huobipro.fetch_trades('LTC/CNY'))

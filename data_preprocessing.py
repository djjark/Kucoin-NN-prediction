import pandas as pd
from collections import deque
from datetime import datetime
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
import os
SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "ETH-USD"
EPOCHS = 10
BATCH_SIZE = 16

def date(var):
    return datetime.utcfromtimestamp(int(var)).strftime('%Y-%m-%d %H(h)%M(m)%S(s)')


NAME = f'{RATIO_TO_PREDICT}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{date(int(time.time()))}'

def preprocess_df(df):
    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic. Those nasty NaNs love to creep in.
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.
    
    buys = []
    sells= []
    
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    random.shuffle(buys)
    random.shuffle(sells)
    
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    
    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
        
    return np.array(X), y



def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


main_df = pd.DataFrame() # begin empty

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
for ratio in ratios:  # begin iteration
    # print(ratio)
    dataset = f'crypto_data/{ratio}.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
# print(main_df.head())  # how did we do??

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))



times = sorted(main_df.index.values)  # get the times
last_5pct = sorted(main_df.index.values)[-int(0.4*len(times))]  # get the last 5% of the times
last1pct = sorted(main_df.index.values)[-int(0.001*len(times))]  # get the last 5% of the times

validation_main_df = main_df[(main_df.index >= last_5pct)]  # make the validation data where the index is in the last 5%
try1 = main_df[(main_df.index >= last1pct)]
main_df = main_df[(main_df.index < last_5pct)]  # now the main_df is all the data up to the last 5%

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)
try_X, try_y = preprocess_df(try1)
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")


def create_model():
    model = tf.keras.models.Sequential([
    layers.LSTM(32, input_shape=(train_x.shape[1:]), return_sequences=True),
    layers.LSTM(32, input_shape=(train_x.shape[1:]), return_sequences=True),
    layers.Dropout(0.2),
    layers.BatchNormalization(),

    layers.LSTM(64, input_shape=(train_x.shape[1:]), return_sequences=True),
    layers.Dropout(0.1),
    layers.BatchNormalization(),

    layers.LSTM(64, input_shape=(train_x.shape[1:])),
    layers.Dropout(0.2),
    layers.BatchNormalization(),

    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),

    layers.Dense(2, activation="softmax"),
  ])
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
    return model

model = create_model()
# model.summary()

# model.summary()
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:::.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy::', verbose=0, save_best_only=True, mode='max')) # saves only the best ones

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)
try_X = np.asarray(try_X)
try_y = np.asarray(try_y)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train model
# history = model.fit(
#     validation_x, validation_y,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     callbacks=[tensorboard, cp_callback],
#     shuffle=False,
# )

model.load_weights(checkpoint_path)
# pred = model.predict(validation_x)
pred = model.predict(try_X)
count=0
for i in pred:
    if i[0]>i[1]:
        print(str(0)+" true: "+str(try_y[count]))
    else:
        print(str(1)+" true: "+str(try_y[count]))
        
    if count>10:
        break
   
    count+=1

    
# Score model
score = model.evaluate(try_X, try_y, verbose=0)
print(str(len(try_X))+" "+str(len(train_x)))
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
# model.save(f"models/{NAME}")





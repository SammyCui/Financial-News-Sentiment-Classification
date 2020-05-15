from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sentiment_analysis_nltk import analyze
from single_stock_analysis import add_prev
from tweets_processing import text_processing
AAPL = pickle.load(open('tweets/AAPL_tweets.pickle', 'rb'))
AAPL_price = pd.read_csv("prices/AAPL.csv")
AAPL_price['Date'] = pd.to_datetime(AAPL_price['Date'])
AAPL['timestamp'] = pd.to_datetime(AAPL['timestamp'])
AAPL.text = AAPL['text'].apply(text_processing)

WMT = pickle.load(open('tweets/WMT_tweets.pickle', 'rb'))
WMT_price = pd.read_csv("prices/WMT.csv")
WMT_price['Date'] = pd.to_datetime(WMT_price['Date'])
WMT['timestamp'] = pd.to_datetime(WMT['timestamp'])
WMT['text'].apply(text_processing)

AMZN = pickle.load(open("tweets/AMZN_tweets.pickle", 'rb'))
AMZN_price = pd.read_csv("prices/AMZN.csv")
AMZN_price['Date'] = pd.to_datetime(AAPL_price['Date'])
AMZN['timestamp'] = pd.to_datetime(AMZN['timestamp'])
AMZN['text'].apply(text_processing)
model = Sequential()
model.add(Embedding(3, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

analyzed = analyze(AAPL)
analyzed = analyzed.groupby('timestamp').mean()

AAPL_price.drop(['Open', 'High', 'Low','Volume', 'Adj Close'], axis = 1, inplace=True)
AAPL_joined = AAPL_price.set_index("Date").join(analyzed).reset_index()
AAPL_joined['label'] = AAPL_joined['Close'] - AAPL_joined['Close'].shift(1)
AAPL_joined['label'] = AAPL_joined['label'].apply(lambda x: 0 if x <0 else 1)
AAPL_joined['label'] = AAPL_joined['label'].shift(-1)
AAPL_joined.dropna(inplace = True)

AMZN_analyzed = analyze(AMZN)
AMZN_analyzed = AMZN_analyzed.groupby('timestamp').mean()

AMZN_price.drop(['Open', 'High', 'Low','Volume', 'Adj Close'], axis = 1, inplace=True)
AMZN_joined = AMZN_price.set_index("Date").join(AMZN_analyzed).reset_index()
AMZN_joined['label'] = AMZN_joined['Close'] - AMZN_joined['Close'].shift(1)
AMZN_joined['label'] = AMZN_joined['label'].apply(lambda x: 0 if x <0 else 1)
AMZN_joined['label'] = AMZN_joined['label'].shift(-1)
AMZN_joined.dropna(inplace = True)

WMT_analyzed = analyze(WMT)
WMT_analyzed = WMT_analyzed.groupby('timestamp').mean()

WMT_price.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True)
WMT_joined = WMT_price.set_index("Date").join(WMT_analyzed).reset_index()
WMT_joined['label'] = WMT_joined['Close'] - WMT_joined['Close'].shift(1)
WMT_joined['label'] = WMT_joined['label'].apply(lambda x: 0 if x < 0 else 1)
WMT_joined['label'] = WMT_joined['label'].shift(-1)
WMT_joined.dropna(inplace = True)

data = pd.concat([WMT_joined, AMZN_joined, AAPL_joined]).reset_index(drop=True).drop('Date', axis = 1)

X = data[['text_positive', 'text_negative', 'text_neutral', 'text_compound']]
y = data['label']

add_prev(AAPL_joined, 'label', 1)
add_prev(AAPL_joined, 'label', 2)
add_prev(AAPL_joined, 'label', 3)
add_prev(AAPL_joined, 'text_positive', 1)
add_prev(AAPL_joined, 'text_positive', 2)
add_prev(AAPL_joined, 'text_positive', 3)
add_prev(AAPL_joined, 'text_negative', 1)
add_prev(AAPL_joined, 'text_negative', 2)
add_prev(AAPL_joined, 'text_negative', 3)
AAPL_joined.dropna(inplace=True)

add_prev(AMZN_joined, 'label', 1)
add_prev(AMZN_joined, 'label', 2)
add_prev(AMZN_joined, 'label', 3)
add_prev(AMZN_joined, 'text_positive', 1)
add_prev(AMZN_joined, 'text_positive', 2)
add_prev(AMZN_joined, 'text_positive', 3)
add_prev(AMZN_joined, 'text_negative', 1)
add_prev(AMZN_joined, 'text_negative', 2)
add_prev(AMZN_joined, 'text_negative', 3)
AMZN_joined.dropna(inplace=True)

add_prev(WMT_joined, 'label', 1)
add_prev(WMT_joined, 'label', 2)
add_prev(WMT_joined, 'label', 3)
add_prev(WMT_joined, 'text_positive', 1)
add_prev(WMT_joined, 'text_positive', 2)
add_prev(WMT_joined, 'text_positive', 3)
add_prev(WMT_joined, 'text_negative', 1)
add_prev(WMT_joined, 'text_negative', 2)
add_prev(WMT_joined, 'text_negative', 3)
WMT_joined.dropna(inplace=True)

data_prev = pd.concat([WMT_joined, AMZN_joined, AAPL_joined]).reset_index(drop=True).drop('Date', axis = 1)
AAPL_rnn_X = data_prev.drop("label", axis=1).to_numpy()
AAPL_rnn_y = data_prev[['label']].to_numpy()

AAPL_rnn_X = np.reshape(AAPL_rnn_X, (AAPL_rnn_X.shape[0], AAPL_rnn_X.shape[1], 1))

from keras.layers import LSTM
model2 = Sequential()
model2.add(LSTM(units=50, return_sequences=True, input_shape= (AAPL_rnn_X.shape[1], 1)))
model2.add(Dense(1))
model2.add(Flatten())
model2.add(Dense(1,activation = 'sigmoid'))

from keras.optimizers import SGD
opt = SGD(lr = 0.0001)
model2.compile(loss='mean_squared_error', optimizer= "Adamax",
              metrics=['acc'])
history2 = model2.fit(AAPL_rnn_X, AAPL_rnn_y,
                    epochs=10,
                    batch_size=10,
                    validation_split=0.2)

acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
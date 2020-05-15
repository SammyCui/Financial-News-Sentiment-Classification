from sentiment_analysis_nltk import analyze
from tweets_processing import text_processing_bow, bag_of_words
import pandas as pd
import pickle


AAPL = pickle.load(open('tweets/AAPL_tweets.pickle', 'rb'))
WMT = pickle.load(open('tweets/WMT_tweets.pickle', 'rb'))
AMZN = pickle.load(open("tweets/AMZN_tweets.pickle", 'rb'))
FB = pickle.load(open("tweets/FB_tweets.pickle", 'rb'))

AAPL_price = pd.read_csv("prices/AAPL.csv")
WMT_price = pd.read_csv("prices/WMT.csv")
AMZN_price = pd.read_csv("prices/AMZN.csv")
FB_price = pd.read_csv("prices/FB.csv")



AAPL_bow = bag_of_words(AAPL)
WMT_bow = bag_of_words(WMT)
AMZN_bow = bag_of_words(AMZN)
FB_bow = bag_of_words(FB)

AAPL_bow['timestamp'] = pd.to_datetime(AAPL_bow['timestamp'])
WMT_bow['timestamp'] = pd.to_datetime(WMT_bow['timestamp'])
AMZN_bow['timestamp'] = pd.to_datetime(AMZN_bow['timestamp'])
FB_bow['timestamp'] = pd.to_datetime(FB_bow['timestamp'])

AAPL_bow = AAPL_bow.groupby('timestamp').sum()
WMT_bow = WMT_bow.groupby('timestamp').sum()
AMZN_bow = AMZN_bow.groupby('timestamp').sum()
FB_bow = FB_bow.groupby('timestamp').sum()

print(AAPL_bow.head())

AAPL_joined = AAPL_price.set_index("Date").join(AAPL_bow).reset_index()
AAPL_joined['label'] = AAPL_joined['Close'] - AAPL_joined['Close'].shift(1)
AAPL_joined['label'] = AAPL_joined['label'].apply(lambda x: 0 if x <0 else 1)
AAPL_joined['label'] = AAPL_joined['label'].shift(-1)
AAPL_joined.dropna(inplace = True)


AMZN_price.drop(['Open', 'High', 'Low','Volume', 'Adj Close'], axis = 1, inplace=True)
AMZN_joined = AMZN_price.set_index("Date").join(AMZN_bow).reset_index()
AMZN_joined['label'] = AMZN_joined['Close'] - AMZN_joined['Close'].shift(1)
AMZN_joined['label'] = AMZN_joined['label'].apply(lambda x: 0 if x <0 else 1)
AMZN_joined['label'] = AMZN_joined['label'].shift(-1)
AMZN_joined.dropna(inplace = True)


WMT_price.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True)
WMT_joined = WMT_price.set_index("Date").join(WMT_bow).reset_index()
WMT_joined['label'] = WMT_joined['Close'] - WMT_joined['Close'].shift(1)
WMT_joined['label'] = WMT_joined['label'].apply(lambda x: 0 if x < 0 else 1)
WMT_joined['label'] = WMT_joined['label'].shift(-1)
WMT_joined.dropna(inplace = True)



FB_price.drop(['Open', 'High', 'Low', 'Volume', 'Adj Close'], axis=1, inplace=True)
FB_joined = FB_price.set_index("Date").join(FB_bow).reset_index()
FB_joined['label'] = FB_joined['Close'] - FB_joined['Close'].shift(1)
FB_joined['label'] = FB_joined['label'].apply(lambda x: 0 if x < 0 else 1)
FB_joined['label'] = FB_joined['label'].shift(-1)
FB_joined.dropna(inplace = True)


data = pd.concat([WMT_joined, AMZN_joined, AAPL_joined, FB_joined]).reset_index(drop=True).drop('Date', axis = 1)

print(data.head())
import pickle
import pandas as pd
from sentiment_analysis_nltk import analyze
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import stats
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from tweets_processing import text_processing


# A smoother that makes graph look smoother.
def smoothen(l, size):
    l_smoothen = []
    for i in range(len(l)):
        if i < size - 1:
            window = l[:i+1]
            average = sum(window)/(i+1)
            l_smoothen.append(average)
        else:
            window = l[i - size + 1: i + 1]
            average = sum(window)/ size
            l_smoothen.append(average)
    return l_smoothen

# Append previous days' values.
def add_prev(df, col, days, inplace = True):
    if inplace:
        df['prev_{}_{}'.format(col, str(days))] = df[col].shift(days)
    else:
        new_df = df.copy()
        new_df['prev_{}_{}'.format(col,days)] = new_df[col].shift(days)
        return new_df

nasdaq = pickle.load(open('tweets/nasdaq_tweets.pickle', 'rb'))
nasdaq_price = pd.read_csv("prices/NDAQ.csv")
nasdaq_price['Date'] = pd.to_datetime(nasdaq_price['Date'])
nasdaq['timestamp'] = pd.to_datetime(nasdaq['timestamp'])
nasdaq.text = nasdaq['text'].apply(text_processing)

sentiment_score_nasdaq = analyze(nasdaq).groupby('timestamp').mean()
x_nasdaq = sentiment_score_nasdaq.index
y_nasdaq_negative = sentiment_score_nasdaq.text_negative
y_nasdaq_list = pd.Series.tolist(y_nasdaq_negative)

y_nasdaq_average = smoothen(y_nasdaq_list, 4)
fig, axs = plt.subplots(2)
fig.suptitle('nasdaq_stock_price vs nasdaq_text_negative')
axs[0].plot(x_nasdaq, y_nasdaq_average)
axs[1].plot(nasdaq_price.Date, nasdaq_price.Close)
axs[0].tick_params(axis = "x", which = "both", bottom = False, top = False, labelbottom = False)
plt.xticks(rotation = 30)
plt.show()

y_nasdaq_compound = sentiment_score_nasdaq.text_compound
y_nasdaq_list = pd.Series.tolist(y_nasdaq_compound)
y__nasdaq_average = smoothen(y_nasdaq_list, 4)
fig, axs = plt.subplots(2)
fig.suptitle('nasdaq_stock_price vs nasdaq_text_compound')
axs[0].plot(x_nasdaq, y_nasdaq_average)
axs[1].plot(nasdaq_price.Date, nasdaq_price.Close)
axs[0].tick_params(axis = "x", which = "both", bottom = False, top = False, labelbottom = False)
plt.xticks(rotation = 30)
plt.show()
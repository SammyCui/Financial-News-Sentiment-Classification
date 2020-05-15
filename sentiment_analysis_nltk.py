from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import os
import pickle
import nltk
import ssl
from collections import *
import copy

# pd dataframe fully displayed
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# nltk.download('punkt')

# nltk.download('lexicon') lookup error
# nltk downloader local certification error. Cannot connect to server
# use ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('vader_lexicon')


BASE_PATH = os.path.join(os.getcwd(),'..','data')

sid = SentimentIntensityAnalyzer()

def analyze(df):

    sent = {}
    pos = []
    neg = []
    com = []
    neu = []
    print(len(df))
    i = 0
    for date, data in df.iterrows():
        i+=1
        sent[i] = sid.polarity_scores(str(data.text))
        pos.append(sent[i]['pos'])
        neg.append(sent[i]['neg'])
        neu.append(sent[i]['neu'])
        com.append(sent[i]['compound'])

    analyzed_df = pd.DataFrame(df[['text', 'timestamp']])

    analyzed_df['text_positive'] = pos
    analyzed_df['text_negative'] = neg
    analyzed_df['text_neutral'] = neu
    analyzed_df['text_compound'] = com
    return analyzed_df


if __name__ == "__main__":
    # new words and values

    lm_wordlist = pd.ExcelFile('LoughranMcDonald_SentimentWordLists_2018.xlsx')
    positive = pd.read_excel(lm_wordlist, 'Positive', header= None, names= "P")
    positive["score"] = 1
    negative = pd.read_excel(lm_wordlist, 'Negative', header = None, names = "N")
    negative["score"] = -1
    word_dict, negative_dict = dict(zip(positive["P"], positive["score"])), dict(zip(negative["N"], negative["score"]))
    word_dict.update(negative_dict)
    print(word_dict)
    financial_words = {
        'crushes': 10,
        'misses': -5,
        'trouble': -10,
        'falls': -10,
        'increase' : 5,
        'decrease' : -5,
        'growth' : 5,
        'risk' : -5

    }

    Analyzer = SentimentIntensityAnalyzer()
    Analyzer.lexicon.update(financial_words)

    sentence = 'It had a fall. '
    tokenized_sentence = nltk.word_tokenize(sentence)
    pos_word_list = []
    neu_word_list = []
    neg_word_list = []

    for word in tokenized_sentence:
        if (Analyzer.polarity_scores(word)['compound']) >= 0.1:
            pos_word_list.append(word)
        elif (Analyzer.polarity_scores(word)['compound']) <= -0.1:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)

    print('Positive:', pos_word_list)
    print('Neutral:', neu_word_list)
    print('Negative:', neg_word_list)
    score = Analyzer.polarity_scores(sentence)
    print('\nScores:', score)



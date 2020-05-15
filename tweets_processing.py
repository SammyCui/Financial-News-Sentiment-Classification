from string import punctuation
import re
from nltk.corpus import stopwords
import pandas as pd
import pickle
import nltk
import heapq
import numpy as np
from collections import *
import copy

# pd.set_option('display.max_rows', None)
# nltk.download('stopwords')


symbols = [symbol.lower() for symbol in pd.read_csv('symbols.csv').Symbol]


# remove URLs in tweets
def remove_junks(words):
    return re.sub(r"http\S+|\S*@\S+|videospic\S+|pic\S+|#\S+|\d+|\S+\.\S+|\$\S+", "", words)


# convert to lower case
def to_lowercase(words):
    new_words = ''
    for word in words:
        new_word = word.lower()
        new_words += new_word
    return new_words


# remove punctuations
def remove_punctuation(sentence):
    new_sentence = ''
    for word in sentence:
        if word not in punctuation:
            new_sentence += word
    return new_sentence


# remove stopwords
def remove_stopwords(sentence):
    sentence_list = sentence.split(" ")
    new_sentence = ''
    for word in sentence_list:
        if word not in stopwords.words('english'):
            new_sentence += word + " "
    return new_sentence

def remove_tags(sentence):
    sentence_list = sentence.split(" ")
    new_sentence = ""

    for word in sentence_list:
        if (not word.__contains__("$")) and word.strip('()') not in symbols:
            new_sentence += word + " "
    return new_sentence



def remove_backslash(sentence):
    return sentence.replace("\n", "")


def text_processing(sentence):

    # Having uppercase and some punctuation can make the sentiments stronger.
    # sentence = remove_punctuation(sentence)
    # sentence = to_lowercase(sentence)
    # sentence = remove_stopwords(sentence)
    sentence = remove_tags(sentence)
    sentence = remove_backslash(sentence)
    sentence = remove_junks(sentence)
    return sentence

def text_processing_bow(sentence):

    sentence = remove_punctuation(sentence)
    sentence = to_lowercase(sentence)
    sentence = remove_stopwords(sentence)

    sentence = remove_tags(sentence)
    sentence = remove_backslash(sentence)
    sentence = remove_junks(sentence)
    return sentence



def bag_of_words(df):
    df_bow = df.copy(deep = True)
    df_bow['text'] = df_bow['text'].apply(text_processing_bow)
    word2count = {}
    for data in df_bow.text:
        words = nltk.word_tokenize(data)
        for word in words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1

    freq_words = heapq.nlargest(150, word2count, key=word2count.get)
    freq_dict = dict(zip(freq_words, [0 for i in range(len(freq_words))]))


    def tokenize(sentence):
        vector = copy.deepcopy(freq_dict)
        for word in sentence:
            if word in freq_dict:
                vector[word] += 1

        return vector.values()

    df_bow['text'] = df_bow['text'].apply(tokenize)
    return df_bow

if __name__ == "__main__":
    AAPL = pickle.load(open('tweets/AAPL_tweets.pickle', 'rb'))
    bow = bag_of_words(AAPL)


import requests
from twitterscraper import query_tweets
import datetime as dt
import pandas as pd
import json, codecs
import numpy as np
import pickle
from tweet_scraper import tweets_scraper
from tweets_processing import text_processing


# To check linear relation between sentiment and stock price
# Pearson Correlation Coefficient

# -------------------- scrape AAPL apple tweets ----------------------- #
start_date = dt.date(2018, 1, 2)
end_date = dt.date(2020, 3, 10)
keyword = "@FinancialTimes"

tweets = tweets_scraper(start_date, end_date, keyword, 1000)
tweets['text'].apply(text_processing)
print(tweets)
pickle_out = open("FinancialTimes_tweets.pickle", "wb")
pickle.dump(tweets, pickle_out)
pickle_out.close()


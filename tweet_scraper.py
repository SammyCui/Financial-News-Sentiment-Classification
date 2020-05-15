from twitterscraper import query_tweets
import datetime as dt
import pandas as pd

def tweets_scraper(begin_date, end_date, keyword, limit, lang = "en"):
    """

    The twitterscraper API doesn't return tweets from everyday, if the time period is long.
    Do one day at a time.

    :param begin_date: begin date of scraping
    :param end_date: end date of scraping
    :param keyword: search keyword
    :param limit: limit on the number of tweets scraping each day.
    :param lang: default = english
    :return: dataframe
    """

    def tweets_scraper_inner(begin_date, end_date, keyword, limit, lang = "en"):
        """
        Using the twitterscraper API. Github source: https://github.com/taspinar/twitterscraper
        return: dataframe.
        """
        tweets = query_tweets(keyword, begindate=begin_date, enddate=end_date, limit=limit, lang=lang)
        text, timestamp, likes, retweets, replies = [], [], [], [], []

        for tweet in tweets:
            text.append(tweet.text)
            timestamp.append(tweet.timestamp)
            likes.append(tweet.likes)
            retweets.append(tweet.retweets)
            replies.append(tweet.replies)

        tweets = pd.DataFrame({"text": text, "timestamp": timestamp, "likes": likes, "retweets": retweets, "replies": replies})

        # Don't need the exact h-m-s, cast it to date object.
        tweets['timestamp'] = tweets['timestamp'].apply(lambda x: str(x.date()))
        return tweets

    delta = dt.timedelta(days=1)
    tweets_collection = pd.DataFrame()
    while begin_date <= end_date:
        begin = begin_date
        begin_date += delta
        end = begin_date
        print(begin, end)
        daily_tweets = tweets_scraper_inner(begin_date=begin, end_date=end, keyword=keyword, limit=limit)
        tweets_collection = tweets_collection.append(daily_tweets)
    return tweets_collection













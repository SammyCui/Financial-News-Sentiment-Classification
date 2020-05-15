from twitterscraper import query_tweets_from_user
import pandas as pd

keyword = "@FinancialTimes"
raw_tweets = query_tweets_from_user(keyword, limit=100000000)
text, timestamp, likes, retweets, replies = [], [], [], [], []

for tweet in raw_tweets:
    text.append(tweet.text)
    timestamp.append(tweet.timestamp)
    likes.append(tweet.likes)
    retweets.append(tweet.retweets)
    replies.append(tweet.replies)

tweets = pd.DataFrame({"text": text, "timestamp": timestamp, "likes": likes, "retweets": retweets, "replies": replies})

# Don't need the exact h-m-s, cast it to date object.
tweets['timestamp'] = tweets['timestamp'].apply(lambda x: str(x.date()))
print(tweets.timestamp.unique())
import sys
import json
import time
import re
import requests
import nltk
import argparse
import logging
import string
import pandas as pd
import numpy
try:
    import urllib.parse as urlparse
except ImportError:
    import urlparse
from tweepy.streaming import StreamListener
from tweepy import API, Stream, OAuthHandler, TweepError
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from random import randint, randrange
from datetime import datetime
from newspaper import Article, ArticleException
import tweepy
consumer_key = "mF4MgflWQgKAzEto8SHgNWWO9"
consumer_secret = "eFmedDj6svNOSuSH6EsNEFDwa3J285qyIcZK3wx8WrnR26Rejp"
access_token = "734186230414381057-y3P2aBvC9Yc2wsspJvhU0H148yOGgMD"
access_token_secret = "7kSE3yZkUdgZikFzgBnTOFiqsHLUso928dfP1mPnACKzZ"
prev_time = time.time()
sentiment_avg = [0.0,0.0,0.0]
name_data=[]
location_data=[]
language_data=[]
friends_data=[]
followers_data=[]
statuses_data=[]
dat_datae=[]
message_data=[]
tweet_id_data=[]
polarity_data=[]
subjectivity_data=[]
sentiment_data=[]
date_data=[]
datenews_data=[]
locationnews_data=[]
messagenews_data=[]
polaritynews_data=[]
subjectivitynews_data=[]
sentimentnews_data=[]
text_data=[]
twitter_feeds = ["@andybiotech","@bradloncar","@zbiotech","@bradloncar","@adamfeuerstein","@VikramKhanna","@cnbc", "@benzinga", "@stockwits",
                 "@Newsweek", "@WashingtonPost", "@breakoutstocks", "@bespokeinvest",
                 "@WSJMarkets", "@stephanie_link", "@nytimesbusiness", "@IBDinvestors",
                 "@WSJDealJournal", "@jimcramer", "@TheStalwart", "@TruthGundlach",
                 "@Carl_C_Icahn", "@ReformedBroker", "@bespokeinvest", "@stlouisfed",
                 "@muddywatersre", "@mcuban", "@AswathDamodaran", "@elerianm",
                 "@MorganStanley", "@ianbremmer", "@GoldmanSachs", "@Wu_Tang_Finance",
                 "@Schuldensuehner", "@NorthmanTrader", "@Frances_Coppola",
                 "@BuzzFeed","@nytimes"]
tweet_ids = []
def clean_text(text):
    # clean up text
    text = text.replace("\n", " ")
    text = re.sub(r"https?\S+", "", text)
    text = re.sub(r"&.*?;", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("RT", "")
    text = text.replace(u"…", "")
    text = text.strip()
    return text



def clean_text_sentiment(text):
    # clean up text for sentiment analysis
    text = re.sub(r"[#|@]\S+", "", text)
    text = text.strip()
    return text


def get_sentiment_from_url(text, sentimentURL):
    # get sentiment from text processing website
    payload = {'text': text}

    try:
        #logger.debug(text)
        post = requests.post(sentimentURL, data=payload)
        #logger.debug(post.status_code)
        #logger.debug(post.text)
    except requests.exceptions.RequestException as re:
        raise

    # return None if we are getting throttled or other connection problem
    if post.status_code != 200:
        return None

    response = post.json()

    neg = response['probability']['neg']
    pos = response['probability']['pos']
    neu = response['probability']['neutral']
    label = response['label']

    # determine if sentiment is positive, negative, or neutral
    if label == "neg":
        sentiment = "negative"
    elif label == "neutral":
        sentiment = "neutral"
    else:
        sentiment = "positive"

    return sentiment, neg, pos, neu


def sentiment_analysis(text):
    """Determine if sentiment is positive, negative, or neutral
    algorithm to figure out if sentiment is positive, negative or neutral
    uses sentiment polarity from TextBlob, VADER Sentiment and
    sentiment from text-processing URL
    could be made better :)
    """

    # pass text into TextBlob
    text_tb = TextBlob(text)

    # pass text into VADER Sentiment
    analyzer = SentimentIntensityAnalyzer()
    text_vs = analyzer.polarity_scores(text)

    # determine sentiment from our sources
    if True:
        if text_tb.sentiment.polarity < 0 and text_vs['compound'] <= -0.05:
            sentiment = "negative"
        elif text_tb.sentiment.polarity > 0 and text_vs['compound'] >= 0.05:
            sentiment = "positive"
        else:
            sentiment = "neutral"

    # calculate average polarity from TextBlob and VADER
    polarity = (text_tb.sentiment.polarity + text_vs['compound']) / 2

    # output sentiment polarity
    print("************")
    print("Sentiment Polarity: " + str(round(polarity, 3)))

    # output sentiment subjectivity (TextBlob)
    print("Sentiment Subjectivity: " + str(round(text_tb.sentiment.subjectivity, 3)))

    # output sentiment
    print("Sentiment (algorithm): " + str(sentiment))
    print("Overall sentiment (textblob): ", text_tb.sentiment)
    print("Overall sentiment (vader): ", text_vs)
    print("sentence was rated as ", round(text_vs['neg']*100, 3), "% Negative")
    print("sentence was rated as ", round(text_vs['neu']*100, 3), "% Neutral")
    print("sentence was rated as ", round(text_vs['pos']*100, 3), "% Positive")
    print("************")

    return polarity, text_tb.sentiment.subjectivity, sentiment


def tweeklink_sentiment_analysis(url):
    # get text summary of tweek link web page and run sentiment analysis on it
    try:
        article = Article(url)
        article.download()
        article.parse()
        # check if twitter web page
        if "Tweet with a location" in article.text:
            return None
        article.nlp()
        tokens = article.keywords
        print("Tweet link nltk tokens:", tokens)

        # check for min token length
        if len(tokens) < 5:
            return None
        # check ignored tokens from config
        for t in nltk_tokens_ignored:
            if t in tokens:
                return None
        # check required tokens from config
        tokenspass = False
        tokensfound = 0
        for t in nltk_tokens_required:
            if t in tokens:
                tokensfound += 1
                if tokensfound == nltk_min_tokens:
                    tokenspass = True
                    break
        if not tokenspass:
            return None

        summary = article.summary
        if summary == '':
            return None
        summary_clean = clean_text(summary)
        summary_clean = clean_text_sentiment(summary_clean)
        print("Tweet link Clean Summary (sentiment): " + summary_clean)
        polarity, subjectivity, sentiment = sentiment_analysis(summary_clean)

        return polarity, subjectivity, sentiment

    except ArticleException as e:
        return None




from config import *
def get_all_tweets(screen_name):
#Twitter only allows access to a users most recent 3240 tweets with this method

    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,parser=tweepy.parsers.JSONParser())

    #initialize a list to hold all the tweepy Tweets
    alltweets = []
    print(screen_name)
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=1)
    # print(new_tweets)

    # save most recent tweets
    alltweets.extend(new_tweets)
    # print(alltweets[-1])
    # save the id of the oldest tweet less one
    oldest = alltweets[-1]['id'] - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        # print ("getting tweets before %s" % (oldest))

        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest, )

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1]['id'] - 1

        print("...%s tweets downloaded so far" % (len(alltweets)))

    return alltweets


def extract(data):
    # decode json
    json_str = json.dumps(data)
    dict_data = json.loads(json_str)

    text = dict_data['text']
    if text is None:
        return True

    # grab html links from tweet
    tweet_urls = []

    # clean up tweet text
    textclean = clean_text(text)

    # check if tweet has no valid text
    if textclean == "":
        return True

    # get date when tweet was created
    created_date = time.strftime(
        '%Y-%m-%dT%H:%M:%S', time.strptime(dict_data['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))

    # store dict_data into vars
    screen_name = str(dict_data.get("user", {}).get("screen_name"))
    location = str(dict_data.get("user", {}).get("location"))
    language = str(dict_data.get("user", {}).get("lang"))
    friends = int(dict_data.get("user", {}).get("friends_count"))
    followers = int(dict_data.get("user", {}).get("followers_count"))
    statuses = int(dict_data.get("user", {}).get("statuses_count"))
    text_filtered = str(textclean)
    tweetid = int(dict_data.get("id"))
    text_raw = str(dict_data.get("text"))

    # output twitter data
    # print("\n<------------------------------")
    # print("Tweet Date: " + created_date)
    # print("Screen Name: " + screen_name)
    # print("Location: " + location)
    # print("Language: " + language)
    # print("Friends: " + str(friends))
    # print("Followers: " + str(followers))
    # print("Statuses: " + str(statuses))
    # print("Tweet ID: " + str(tweetid))
    # print("Tweet Raw Text: " + text_raw)
    # print("Tweet Filtered Text: " + text_filtered)

    # create tokens of words in text using nltk
    text_for_tokens = re.sub(
        r"[\%|\$|\.|\,|\!|\:|\@]|\(|\)|\#|\+|(``)|('')|\?|\-", "", text_filtered)
    tokens = nltk.word_tokenize(text_for_tokens)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [w for w in stripped if w.isalpha()]
    # filter out stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # remove words less than 3 characters
    tokens = [w for w in tokens if not len(w) < 3]
    #print("NLTK Tokens: " + str(tokens))

    # check for min token length
    if len(tokens) < 5:
        return True

    # do some checks before adding to elasticsearch and crawling urls in tweet
    if friends == 0 or \
            followers == 0 or \
            statuses == 0 or \
            text == "" or \
            tweetid in tweet_ids:
        return True

    # check ignored tokens from config
    for t in nltk_tokens_ignored:
        if t in tokens:
            return True
    # check required tokens from config
    tokenspass = False
    tokensfound = 0
    for t in nltk_tokens_required:
        if t in tokens:
            tokensfound += 1
            if tokensfound == nltk_min_tokens:
                tokenspass = True
                break
    if not tokenspass:
        return True

    # clean text for sentiment analysis
    text_clean = clean_text_sentiment(text_filtered)

    # check if tweet has no valid text
    if text_clean == "":
        return True

    #print("Tweet Clean Text (sentiment): " + text_clean)

    # get sentiment values
    polarity, subjectivity, sentiment = sentiment_analysis(text_clean)

    # add tweet_id to list
    tweet_ids.append(dict_data["id"])

    # get sentiment for tweet
    if len(tweet_urls) > 0:
        tweet_urls_polarity = 0
        tweet_urls_subjectivity = 0
        for url in tweet_urls:
            res = tweeklink_sentiment_analysis(url)
            if res is None:
                continue
            pol, sub, sen = res
            tweet_urls_polarity = (tweet_urls_polarity + pol) / 2
            tweet_urls_subjectivity = (tweet_urls_subjectivity + sub) / 2
            if sentiment == "positive" or sen == "positive":
                sentiment = "positive"
            elif sentiment == "negative" or sen == "negative":
                sentiment = "negative"
            else:
                sentiment = "neutral"

        # calculate average polarity and subjectivity from tweet and tweet links
        if tweet_urls_polarity > 0:
            polarity = (polarity + tweet_urls_polarity) / 2
        if tweet_urls_subjectivity > 0:
            subjectivity = (subjectivity + tweet_urls_subjectivity) / 2
    text_data.append(text_clean)
    polarity_data.append(polarity)
    subjectivity_data.append(subjectivity)
    tweet_id_data.append(tweetid)
    message_data.append(text_filtered)
    statuses_data.append(statuses)
    followers_data.append(followers)
    friends_data.append(friends)
    language_data.append(language)
    location_data.append(location)
    name_data.append(screen_name)
    date_data.append(created_date)
    sentiment_data.append(sentiment)

dataframes=[]
for user in twitter_feeds:
    tweets = get_all_tweets(user)
    for tweet in tweets:
        extract(tweet)
    databank = {"author": name_data,
                "location": location_data,
                "language": language_data,
                "friends": friends_data,
                "followers": followers_data,
                "statuses": statuses_data,
                "date": date_data,
                "message": message_data,
                "tweet_id": tweet_id_data,
                "polarity": polarity_data,
                "subjectivity": subjectivity_data,
                "sentiment": sentiment_data,
                "text": text_data}
    twitterdata = pd.DataFrame(databank)
    dataframes.append(twitterdata)
print(dataframes)

pd.set_option('display.max_colwidth', -1)
print(twitterdata)
print(twitterdata['text'])
print(twitterdata.iloc[0])
print(nltk_tokens_required)
twitterdata.to_csv('asdrptwitterdata.csv')

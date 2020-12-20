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
import tweepy as tw
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch
from random import randint, randrange
from datetime import datetime
from newspaper import Article, ArticleException

# import elasticsearch host, twitter keys and tokens
from config import *
consumer_key = "mF4MgflWQgKAzEto8SHgNWWO9"
consumer_secret = "eFmedDj6svNOSuSH6EsNEFDwa3J285qyIcZK3wx8WrnR26Rejp"
access_token = "734186230414381057-y3P2aBvC9Yc2wsspJvhU0H148yOGgMD"
access_token_secret = "7kSE3yZkUdgZikFzgBnTOFiqsHLUso928dfP1mPnACKzZ"
twitter_users_file = './twitteruserids.txt'

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
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)
tweets = tw.Cursor()
tweets
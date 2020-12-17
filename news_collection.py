"""
This code takes news data from finviz.com from the last 5 days and then
enters the data into a spreadsheet after summarizing and analyzing the data
"""

# These are the important imports needed for this project.
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from datetime import datetime, timedelta
import spacy
from textblob import TextBlob
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest


finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['UNH', 'BIIB', 'GILD', 'JNJ', 'PFE']    # Enter the tickers that you would like to scrape from

current_date = datetime.now()
goal = current_date - timedelta(days=4)  # Change the number if you want to change the number of days worth of data

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})   # Opens the finviz url for datascraping
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')     # Finds the html tag of new-table and takes the data from that div
    news_tables[ticker] = news_table



parsed_data = []

for ticker, news_table in news_tables.items():
    """
    This goes through all the table rows of data and then stores the data, url, title, time and ticker 
    in a pandas dateframe.
    """
    for row in news_table.findAll('tr'):
        for a in row.find_all('a', href=True):
           url = a['href']
        title = row.a.text
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        # This code is used to check whether the articles are too old.
        article_date = date
        article_date = article_date.replace('-', '/')
        article_date = article_date.replace('Nov', '11')
        article_date = article_date.replace('Oct', '10')
        article_date = article_date.replace('Dec', '12')
        article_date = article_date.replace('Jan', '1')
        article_date = article_date.replace('Feb', '2')
        article_date = datetime.strptime(article_date, '%m/%d/%y')
        if goal < article_date:
            parsed_data.append([ticker, date, time, title, url])
        else:
            continue

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title', 'url'])


news_articles = []
for i, row in df.iterrows():   # parses through the rows in the pandas dateframe in order to get the url for webscraping
    print(i, row[0], row[1], row[2], row[3], row[4])
    url = row[4]
    url_data = urlparse(url)
    if url_data.netloc == 'www.barrons.com': # barrons has a paywall so if the url is barrons then skip
        continue
    elif '/video/' in url_data.path:  # some links have videos so we skip those links
        continue
    elif url_data.netloc != 'finance.yahoo.com': # if the link is not from yahoo then store temp content.
        news = 'This is not yahoo finance'
        news_articles.append([row[0], row[1], row[2], row[3], row[4], news])
    elif url_data.netloc == 'finance.yahoo.com':
        reqs = Request(url=url, headers={'user-agent': 'my-app'})  # Uses requests module to open url
        data = urlopen(reqs)
        soup = BeautifulSoup(data, features='html.parser')
        news = soup.find('div', attrs={'class': 'caas-body'}).text  # Find the tag caas-body in the html
        news_articles.append([row[0], row[1], row[2], row[3], row[4], news]) # Append date to a table

# Creating a timestamp
now = datetime.now()
now_string = str(now.strftime("%m-%d-%Y_%H_%M_%S"))

# Storing all data into a pandas dataframe and then putting the data in a timestamped excel file
df1 = pd.DataFrame(news_articles, columns=['ticker', 'date', 'time', 'title', 'url', 'news'])


nlp = spacy.load('en_core_web_sm')
stopwords = list(STOP_WORDS)


def text_summarizer(raw_docx):
    raw_text = raw_docx
    docx = nlp(raw_text)
    stopwords = list(STOP_WORDS)
    # Build Word Frequency
# word.text is tokenization in spacy
    word_frequencies = {}
    for word in docx:
        if word.text not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Sentence Tokens
    sentence_list = [ sentence for sentence in docx.sents ]

    # Calculate Sentence Score and Ranking
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if len(sent.text.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

    # Find N Largest
    summary_sentences = nlargest(7, sentence_scores, key=sentence_scores.get)
    final_sentences = [ w.text for w in summary_sentences ]
    summary1 = ' '.join(final_sentences)
    return summary1

news_summary = []

for rows in news_articles:
    text_string = rows[5]
    text_string = str(text_string)
    summary = text_summarizer(text_string)
    summary = str(summary)
    news_summary.append([rows[0], rows[1], rows[2], rows[3], rows[4], summary])


# Creating a timestamp
now = datetime.now()
now_string = str(now.strftime("%m-%d-%Y_%H_%M_%S"))

# Storing all data into a pandas dataframe and then putting the data in a timestamped excel file
df2 = pd.DataFrame(news_summary, columns=['ticker', 'date', 'time', 'title', 'url', 'summary'])



def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

news_sentiment = []

for row in news_summary:
    text = row[5]
    text = str(text)
    if text == 'This is not yahoo finance':
        continue
    else:
        subjectivity = TextBlob(text).sentiment.subjectivity
        polarity = TextBlob(text).sentiment.polarity
        SIA = getSIA(text)
        compound = SIA['compound']
        neg = SIA['neg']
        neu = SIA['neu']
        pos = SIA['pos']
        news_sentiment.append([row[0], row[1], row[2], row[3], row[4], text, subjectivity, polarity, compound, neg, neu,
                           pos])

df3 = pd.DataFrame(news_sentiment, columns=['ticker', 'date', 'time', 'title', 'url', 'summary', 'subjectivity',
                                          'polarity', 'compound', 'negative', 'neutral', 'positive' ])
df3.to_excel(f'{now_string}.xlsx')

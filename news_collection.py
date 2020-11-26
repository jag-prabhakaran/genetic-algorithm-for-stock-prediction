"""
This code takes news data from finviz.com from the last 5 days and then
enters the data into a spreadsheet
V1.0.0 - Can only scrape data from finance.yahoo.com
"""

# These are the important imports needed for this project.
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta


finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['UNH', 'BIIB', 'GILD', 'JNJ', 'PFE']    # Enter the tickers that you would like to scrape from

current_date = datetime.now()
goal = current_date - timedelta(days=5)  # Change the number if you want to change the number of days worth of data

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
df1.to_excel(f'{now_string}.xlsx')
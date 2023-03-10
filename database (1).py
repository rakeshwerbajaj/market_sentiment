#!/usr/bin/env python
# coding: utf-8

# In[117]:


import requests 
from datetime import date
loginurl= ('https://aktien.guide/users/sign_in')
secure_url = ('https://aktien.guide/')

indice_url = 'https://aktien.guide/indices'

sector_url = 'https://aktien.guide/sectors'



# In[118]:


payload = {
    'username': 'falk@fcr.ag',
    'password': 'Gaga2022!"'
}


# In[119]:


r= requests.post(loginurl, data= payload)


# In[120]:


from bs4 import BeautifulSoup


# In[121]:


with requests.session () as s: 
    s.post(loginurl, data=payload)
    r = s.get(indice_url)
    soup = BeautifulSoup(r.content, 'html.parser')
    #print(soup.prettify())


# In[122]:


#FAST 
hrf_list = ["https://aktien.guide" + i.find('a')['href']  for i in soup.find_all('div',{'class':'col-6'})]


# In[123]:


len(hrf_list)


# In[124]:


import requests
import json
import pandas as pd
from datetime import datetime

headers = {
    'authority': 'aktien.guide',
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    # 'cookie': 'LD_T=3b84a2bc-115d-4547-fa42-34f39ba74595; LD_U=https%3A%2F%2Faktien.guide%2F; LD_R=https%3A%2F%2Fwww.google.com%2F; _gcl_au=1.1.830348074.1676998836; _fbp=fb.1.1676998836106.185605629; hubspotutk=0602af223820fd0d00055ee46f1d6211; __hssrc=1; _hjSessionUser_3291522=eyJpZCI6ImE1YTA4NTc2LTI1MzEtNWE5NS05YWE5LTE4MzY3ZmY4ZjRiOSIsImNyZWF0ZWQiOjE2NzY5OTg4MzU5NzIsImV4aXN0aW5nIjp0cnVlfQ==; _gid=GA1.2.735018448.1677504873; _hjAbsoluteSessionInProgress=0; _hjIncludedInSessionSample_3291522=0; _hjSession_3291522=eyJpZCI6ImYwYWU1ZTM1LTU1OTUtNDUxOC1hZWE1LTU5YjQ2NzIyNDcyMyIsImNyZWF0ZWQiOjE2Nzc1Nzg5MzI1NDUsImluU2FtcGxlIjpmYWxzZX0=; _hjIncludedInPageviewSample=1; __hstc=112749147.0602af223820fd0d00055ee46f1d6211.1676998836698.1677578933290.1677581564114.6; _ga=GA1.2.1941339273.1676998836; __hssc=112749147.9.1677581564114; 2591h3.grsf.uuid=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiNzAzMTliYjQtOTQwNC00OTgzLTg2NmMtZGNlNTZiYWNjYTllIiwiaWF0IjoxNjc3NTgyMTE4LCJleHAiOjE2ODExODIxMTh9.JhYLwNZ4H8rMBkj5b-lML1HLmyg5fO0QF0uikr4y5gU; _aktien_guide_session=scc%2BCZm6OhpKxl7nbUtod%2FGmrS5Px7B2OzoVZjm8IewoWAPAHm9WxvKG9EIMTTDQLAQQu%2Bix7yDY%2BFN6hzifp3VH43wqd6JvvfgPBvhvxkPQ9czwKmwvDLY%2BxQfh9mspI0spdOn%2FgsXNOgZ7OyRFLUk4oEFfiyYXuQAgIfYBDhkqqbjH9qmpjqMh0Bs8wF0LWMVnzYPwDoYX9O1%2BzsLW7JevToqKtqC1eR1sN1KhU%2FWD3vsNaeMbaIIQo9yDbVOGv4FIY4%2BuSjwKaz3yf9C0WX47%2FXJ6aqe88WfYYCdvzR51e0%2F0BRDo2pZCJxg9DJSWTIM6jLq0zBGDIuEIbJqyQCqlIl2PWQcffO%2BtAINmXAsaCV1Hsx5Y%2F9AexJ33D6x%2FcwJ%2FMwk09o7yqKXSi3QNYD3HXmrhp0rBPgrtvydejOtuH7TSJfF28rvUuzZ73rxKQ46QLgXSa7gf7o6CW%2FoZDfETp4Pd9YB%2F21xnjYeTZQNrhJzysScm3IiayE%2B2Gs%2BLPO01i5ycQ5NtaCkIQAoo7N0WN8TOaBHUKvCW0BBJYmJXSpsRGb2EMCp%2F57NJMC66NfMcdyfHQs7P--q%2FdIvnGQy%2BS9NHl4--2t%2FsaDY1NScgScfjR40HgQ%3D%3D; _ga_0Z934YT7SE=GS1.1.1677581563.7.1.1677582214.0.0.0; LD_S=1677582214298',
    'referer': 'https://aktien.guide/industries/it-services',
    'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest',
}

params = {
    'authenticity_token': 'q90mOK44T25MnWEuCLNI-TXBV8AMEvrEplVgj971tqaBpfi0zGYeCdw0wUQOgTvjblY4u53e_jG2SqYqWll6iA',
    '_': '1677582214211',
}
list_df = []
for ind in hrf_list:
    q = ind.replace('https://aktien.guide/indices/','')
    response = requests.get(ind+'/results', params=params, headers=headers)
    data = json.loads(response.content)
    df = pd.DataFrame(data['stocks']).fillna('None')
    df['Index'] = [q]*len(df)
    df['datetime'] = [str(date.today())]*len(df)
    list_df.append(df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[125]:



# import mysql.connector 
# from mysql.connector import Error
# server = "localhost" # replace with your server name
# database = "new_schema1" # replace with your database name
# username = "root" # replace with your username
# password = "Kingisking123" # replace with your password

# mydb = mysql.connector.connect(host = server, user = username,password = password,database = database)

# if (mydb):
#     print("database connection successful")
# else:
#     print("Not successful")


# In[126]:


#pip install mysql-connector-python


# In[127]:


# # create_stock_table = """
# # CREATE TABLE allstock (
# #   stockid INT PRIMARY KEY,
# #   stockname VARCHAR(40) NOT NULL,
# #   dividend VARCHAR(40) NOT NULL,
# #   highgrowthinv VARCHAR(40) NOT NULL,
# #   leverman VARCHAR(40) NOT NULL,
# #   index VARCHAR(40);
# #   );
# #  """


# def execute_query(connection, query):
#     cursor = connection.cursor()
#     try:
#         cursor.execute(query)
#         connection.commit()
#         print("Query successful")
#     except Error as err:
#         print(f"Error: '{err}'")
        
# create_stock_table = """
# CREATE TABLE my_table3 (
#   stock_id INT PRIMARY KEY AUTO_INCREMENT,
#   stockname VARCHAR(200),
#   dividend INT ,
#   high_growth INT ,
#   leverman INT ,
#   Indexes VARCHAR(100),
#   stock_date DATE
#   );
#  """
# execute_query(mydb, create_stock_table) # Execute our defined query


# In[128]:


#Deuchland
df_1 = list_df[0].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index','datetime' ]]
df_2= list_df[1].loc[:,['name', 'dividend_score', 'hgi_score','levermann_score','Index','datetime' ]]
df_3= list_df[2].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index','datetime' ]]
df_4= list_df[3].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index','datetime' ]]
df_5= list_df[4].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index','datetime' ]]
df_6= list_df[5].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index','datetime' ]]
df_7 =list_df[7].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index','datetime' ]]
df_14 =list_df[13].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index','datetime' ]]
df_15 = list_df[14].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index','datetime' ]]



# In[129]:


#welt: combination of stoxx50 and eurostock600 and DOW without us stocks

#combine all dataframes
# df_welt = pd.concat([df_7,df_14,df_15],ignore_index=True)
# len(df_welt)

dd1 = df_7.append(df_14, ignore_index=True)
df_welt = dd1.append(df_15, ignore_index=True)


# In[130]:


df_d1  = df_1.append(df_2, ignore_index=True)
df_d2 = df_d1.append(df_3, ignore_index=True)
df_d3 = df_d2.append(df_4, ignore_index=True)
df_deu = df_d3.append(df_5, ignore_index=True) 
df_deu=df_deu.drop_duplicates(subset = "name")
df_deu.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[131]:


# import mysql.connector 
# from mysql.connector import Error
# server = "localhost" # replace with your server name
# database = "new_schema1" # replace with your database name
# username = "root" # replace with your username
# password = "Kingisking123" # replace with your password

# conn = mysql.connector.connect(host = server, user = username,password = password,database = database)

# cursor = conn.cursor()

# for row in df_deu.itertuples():
#     stockname = row[1]
#     dividend = row[2]
#     high_growth = row[3]
#     leverman = row[4]
#     Indexes = row[5]
#     datetime = row[6]
#     pop_stocks = "INSERT INTO `my_table3` ( `stockname`, `dividend`, `high_growth`, `leverman`, `Indexes`,`stock_date`) VALUES (%s, %s, %s, %s, %s, %s)"
#     cursor.execute(pop_stocks,(stockname,dividend,high_growth,leverman,Indexes, datetime))
# conn.commit()    


# In[132]:


# df_deu.head()


# In[ ]:





# In[ ]:





# In[133]:


# def read_query(connection, query):
#     cursor = connection.cursor()
#     result = None
#     try:
#         cursor.execute(query)
#         result = cursor.fetchall()
#         return result
#     except Error as err:
#         print(f"Error: '{err}'")
        
# q1 = """
# SELECT *
# FROM my_table3;
# """

# results = read_query(conn, q1)

# for result in results:
#   print(result)



# In[ ]:





# In[134]:


# # delete table 


# def execute_query(connection, query):
#     cursor = connection.cursor()
#     try:
#         cursor.execute(query)
#         connection.commit()
#         print("Query successful")
#     except Error as err:
#         print(f"Error: '{err}'")
        
# create_stock_table = """
# DROP TABLE my_table2
#  """
# execute_query(mydb, create_stock_table) # Execute our defined query


# In[ ]:





# In[ ]:





# In[ ]:


# #Deuchland
# df_1 = list_df[0].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index' ]]
# df_2= list_df[1].loc[:,['name', 'dividend_score', 'hgi_score','levermann_score','Index' ]]
# df_3= list_df[2].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index' ]]
# df_4= list_df[3].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index' ]]
# df_5= list_df[4].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index' ]]
# df_6= list_df[5].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index' ]]
# df_7 =list_df[7].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index' ]]
# df_14 =list_df[13].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index' ]]
# df_15 = list_df[14].loc[:,['name','dividend_score', 'hgi_score','levermann_score','Index' ]]



# In[356]:


# #welt: combination of stoxx50 and eurostock600 and DOW without us stocks

# #combine all dataframes
# # df_welt = pd.concat([df_7,df_14,df_15],ignore_index=True)
# # len(df_welt)

# dd1 = df_7.append(df_14, ignore_index=True)
# df_welt = dd1.append(df_15, ignore_index=True)


# In[ ]:





# In[ ]:





# In[136]:


df_7['dividend_score'][0]


# In[354]:


df_14['dividend_score'][0]


# In[358]:


df_welt['dividend_score'][0]


# In[362]:


df_welt=df_welt.drop_duplicates(subset = "name")


# In[ ]:





# In[366]:


# df_d1  = df_1.append(df_2, ignore_index=True)
# df_d2 = df_d1.append(df_3, ignore_index=True)
# df_d3 = df_d2.append(df_4, ignore_index=True)
# df_deu = df_d3.append(df_5, ignore_index=True) 
# df_deu=df_deu.drop_duplicates(subset = "name")
# df_deu.head()


# In[77]:


len(df_deu)


# In[137]:


df_master = df_deu.append(df_welt, ignore_index=True)
df_master=df_master.drop_duplicates(subset = "name")
df_master.head()


# In[138]:


#start from here addition and 
df_master['points'] = df_master['dividend_score'] + df_master['hgi_score'] + df_master['levermann_score']
df_master['dividend_score'][0] 


# In[139]:


df_m1 =  df_master.sort_values(by=['points'])
df_m1 = df_m1[df_m1['points']>20]


# In[140]:


def fetch_sentiment_using_SIA(text):
    sid = SentimentIntensityAnalyzer()
    polarity_scores = sid.polarity_scores(text)
    return 'neg' if polarity_scores['neg'] > polarity_scores['pos'] else 'pos'

def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, text)
    for i in r:
        text = re.sub(i, '', text)
    
    return text 

def getTicker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "Germany"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code


# In[152]:


# symbols = []
# for k in df_m1['name']:
#     if k=='Klöckner & Co':
#         k='Kloeckner'
#     if k == "Hannover Rück":
#         k = "Hannover"
#     if k == "Hermès (Hermes International)":
#         k = 'Hermes International SA'
#     if k == "DWS (Deutsche Asset Management)":
#         k = 'DWS Group GmbH & Co. KGaA'
#     if k == "Neste Oil":
#         k = 'Neste Oyj'
#     if k == "KSB vz":
#         k = 'KSB SE & Co. KGaA'
#     if k == "BMW Vz":
#         k = 'Bayerische Motoren Werke Aktiengesellschaft'  
#     if k == "Mercedes-Benz Group (Daimler)":
#         k = 'Mercedes-Benz Group AG'     
#     if k == "LOréal":
#         k = 'Loreal'
#     if k == "Volkswagen VZ":
#         k = 'Volkswagen AG'
        
    
#     if k == "A.P. Møller-Mærsk":
#         k = 'A.P.'
    
#     if k == "Kühne + Nagel International":
#         k = 'Kuehne + Nagel International AG'
        
#     print('k : '+ k)
#     name = getTicker(k)
#     print('name : '+ name)
#     symbols.append(name)

# df_m1['symbols'] = symbols


# In[ ]:





# In[143]:


len(df_m1)


# In[144]:


df_m1 = df_m1.iloc[:,[0,7,1,2,3,6,4,5]]


# In[145]:


df_m1


# In[146]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import time
import string
import warnings
import yfinance as yf
import pandas as pd 
import pandas_ta as pta
import requests
import statistics as st
# for all NLP related operations on text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.classify import NaiveBayesClassifier
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# To consume Twitter's API
import tweepy
from tweepy import OAuthHandler 

# To identify the sentiment of text
# from textblob import TextBlob
# from textblob.sentiments import NaiveBayesAnalyzer
# from textblob.np_extractors import ConllExtractor

# For Deploy
import pickle
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.pipeline import make_pipeline
from nltk.tokenize import RegexpTokenizer
from dash import dash_table, Dash 
import pandas as pd

# ignoring all the warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# downloading stopwords corpus
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('conll2000')
nltk.download('brown')
stopwords = set(stopwords.words("english"))

# for showing all the plots inline
get_ipython().run_line_magic('matplotlib', 'inline')


# In[147]:


consumer_key = 'gqW9birtLMLNRTjgv1BOre61N'
consumer_secret = 'vnKkNrt8BOoAHqSeKPYxQ698e2vYCuAFzgSexBmZ7ZCjHl0LKq'
access_token = '3349806612-rV18nciBtRlU5L9b19YjKJuZVtqK1fqgbgGimg0'
access_token_secret = 'zEPMLK1l7pq8S6TWnkM4a6udCkIB77YjKeK3R6UduhRwq'


# In[148]:


class TwitterClient(object): 
    def __init__(self): 
        #Initialization method. 
        try: 
            # create OAuthHandler object 
            auth = OAuthHandler(consumer_key, consumer_secret) 
            # set access token and secret 
            auth.set_access_token(access_token, access_token_secret) 
            # create tweepy API object to fetch tweets 
            # add hyper parameter 'proxy' if executing from behind proxy "proxy='http://172.22.218.218:8085'"
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            
        except tweepy.errors.TweepyException as e:
            print(f"Error: Tweeter Authentication Failed - \n{str(e)}")

    def get_tweets(self, query, maxTweets = 1000):
        #Function to fetch tweets. 
        # empty list to store parsed tweets 
        tweets = [] 
        sinceId = None
        max_id = -1
        tweetCount = 0
        tweetsPerQry = 100

        while tweetCount < maxTweets:
            try:
                if (max_id <= 0):
                    if (not sinceId):
                        new_tweets = self.api.search_tweets(q=query, count=tweetsPerQry)
                    else:
                        new_tweets = self.api.search_tweets(q=query, count=tweetsPerQry,
                                                since_id=sinceId)
                else:
                    if (not sinceId):
                        new_tweets = self.api.search_tweets(q=query, count=tweetsPerQry,
                                                max_id=str(max_id - 1))
                    else:
                        new_tweets = self.api.search_tweets(q=query, count=tweetsPerQry,
                                                max_id=str(max_id - 1),
                                                since_id=sinceId)
                if not new_tweets:
                    print("No more tweets found")
                    break

                for tweet in new_tweets:
                    parsed_tweet = {} 
                    parsed_tweet['tweets'] = tweet.text 

                    # appending parsed tweet to tweets list 
                    if tweet.retweet_count > 0: 
                        # if tweet has retweets, ensure that it is appended only once 
                        if parsed_tweet not in tweets: 
                            tweets.append(parsed_tweet) 
                    else: 
                        tweets.append(parsed_tweet) 
                        
                tweetCount += len(new_tweets)
                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id

            except tweepy.errors.TweepyException as e:
                # Just exit if any error
                print("Tweepy error : " + str(e))
                break
        
        return pd.DataFrame(tweets)


# In[149]:


def fetch_sentiment_using_SIA(text):
    sid = SentimentIntensityAnalyzer()
    polarity_scores = sid.polarity_scores(text)
    return 'neg' if polarity_scores['neg'] > polarity_scores['pos'] else 'pos'

def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, text)
    for i in r:
        text = re.sub(i, '', text)
    
    return text 

def getTicker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "Germany"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code


# In[ ]:





# In[ ]:





# In[150]:


sentiment_results_of_companies = []
for i in df_m1['name']: 
    print(i, "company is searching on twitter")

    twitter_client = TwitterClient()

    # calling function to get tweets
    tweets_df = twitter_client.get_tweets(i, maxTweets=100)
    print(f'tweets_df Shape - {tweets_df.shape}')
    if (tweets_df.shape[1]==0):
        tweets_df["tweets"] = ["No_tweet_for_this_company"]


    sentiments_using_SIA = tweets_df.tweets.apply(lambda tweet: fetch_sentiment_using_SIA(tweet))
    pd.DataFrame(sentiments_using_SIA.value_counts())

    tweets_df['sentiment'] = sentiments_using_SIA
    #tweets_df.head()

    tweets_df['tidy_tweets'] = np.vectorize(remove_pattern)(tweets_df['tweets'], "@[\w]*: | *RT*")
    #tweets_df.head(10)

    cleaned_tweets = []

    for index, row in tweets_df.iterrows():
        # Here we are filtering out all the words that contains link
        words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]
        cleaned_tweets.append(' '.join(words_without_links))

    tweets_df['tidy_tweets'] = cleaned_tweets
    tweets_df = tweets_df.dropna()
    #tweets_df.head(10)
    sentiments_using_SIA2 = tweets_df.tidy_tweets.apply(lambda tweet: fetch_sentiment_using_SIA(tweet))
    pd.DataFrame(sentiments_using_SIA2.value_counts())
    
    positive_tweets = []
    negative_tweets = []
    for j in sentiments_using_SIA2:
        if j == 'pos':
            positive_tweets.append(j)
        else:
            negative_tweets.append(j)
    
    ng_tweets = len(negative_tweets)
    if (ng_tweets == 0):
        ng_tweets = ng_tweets+1 
    if (len(positive_tweets)*0.80)/(ng_tweets) > 0:
        result ='postive'
    else:
        result ='negative'
    
    sentiment_results_of_companies.append(result)
   
        
df_sentiment = pd.DataFrame(sentiment_results_of_companies)
df_sentiment['sentiment'] = df_sentiment
df_sentiment = df_sentiment.iloc[: , 1:]



# In[151]:


symbols = []
for k in df_m1['name']:
    if k=='Klöckner & Co':
        k='Kloeckner'
    if k == "Hannover Rück":
        k = "Hannover"
    if k == "Hermès (Hermes International)":
        k = 'Hermes International SA'
    if k == "DWS (Deutsche Asset Management)":
        k = 'DWS Group GmbH & Co. KGaA'
    if k == "Neste Oil":
        k = 'Neste Oyj'
    if k == "KSB vz":
        k = 'KSB SE & Co. KGaA'
    if k == "BMW Vz":
        k = 'Bayerische Motoren Werke Aktiengesellschaft'  
    if k == "Mercedes-Benz Group (Daimler)":
        k = 'Mercedes-Benz Group AG'     
    if k == "LOréal":
        k = 'Loreal'
    if k == "Volkswagen VZ":
        k = 'Volkswagen AG'
        
    
    if k == "A.P. Møller-Mærsk":
        k = 'A.P.'
    
    if k == "Kühne + Nagel International":
        k = 'Kuehne + Nagel International AG'
        
    print('k : '+ k)
    name = getTicker(k)
    print('name : '+ name)
    symbols.append(name)

df_m1['symbols'] = symbols


# In[154]:


# rsi_inclusion

rsi_list = []

for p in df_m1['symbols']:
    ticker = yf.Ticker(p)
    df_prices = ticker.history(interval='1d')
    df1 = pta.rsi(df_prices['Close'], length = 14)
    rsi_list.append(df1[-1])
    
df_m1['RSI'] = rsi_list

df_rsi_limit1 = df_m1[df_m1['RSI'].between(30,70)]
df_rsi_limit2 = df_m1[df_m1['RSI'].between(0,30)]
df_rsi_limit3 = df_m1[df_m1['RSI'].between(70,100)]

df_rsi_limit1['RSI_signal'] = 'Hold'
df_rsi_limit2['RSI_signal'] = 'Buy_check_price'
df_rsi_limit3['RSI_signal'] = 'Sell_with_price'

df3 = pd.concat([df_rsi_limit1,df_rsi_limit2])
df_rsi_signal = pd.concat([df3,df_rsi_limit3])


# In[155]:


df_m1 = df_rsi_signal # this dataframe is populated.


# In[156]:



# Falk stratgey drop or increase in percentage considering last six weeks average

falk_list = []

for i in df_m1['symbols']:
    ticker = yf.Ticker(i)
    df_prices = ticker.history(interval='1wk', period='max')
    df1  =  df_prices.tail(6)
    final = st.mean(df1['Close'])
    init = df1['Close'][-1]
    falk_result = ((init-final)/init)*100
    falk_list.append(falk_result)

df_m1['Inc/Dec_6_weeks_av_by_Falk%']= falk_list


#main_dataframe = pd.concat([df_m1, df_sentiment], axis=0)


# In[157]:


df_m1['Sentiments'] = sentiment_results_of_companies


# In[158]:


#pip install dash_auth


# In[ ]:





# In[ ]:





# In[ ]:





# In[159]:


from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_auth

app = Dash(__name__)

auth = dash_auth.BasicAuth(app, {
    'Falk': 'Falk'
})

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
        dcc.Tab(label='Master', value='tab-master', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Deutschland', value='tab-deu', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='DAX', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='MDAX', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='SDAX', value='tab-3', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='TecDAX', value='tab-4', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='CDAX', value='tab-5', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Scale', value='tab-6', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Welt', value='tab-welt', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='DOW', value='tab-7', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='EuroStoxx50', value='tab-14', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Stocks50:600', value='tab-15', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Sorted Results', value='tab-s', style=tab_style, selected_style=tab_selected_style)

    ], style=tabs_styles),
    html.Div(id='tabs-content-inline')
])

@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dash_table.DataTable(df_1.to_dict('records'),[{"name": i, "id": i} for i in df_1.columns], id='tbl')
        ])
    elif tab == 'tab-2':
        return html.Div([
            dash_table.DataTable(df_2.to_dict('records'),[{"name": i, "id": i} for i in df_2.columns], id='tb2')
        ])
    elif tab == 'tab-3':
        return html.Div([
            dash_table.DataTable(df_3.to_dict('records'),[{"name": i, "id": i} for i in df_3.columns], id='tb3')
        ])
    elif tab == 'tab-4':
        return html.Div([
            dash_table.DataTable(df_4.to_dict('records'),[{"name": i, "id": i} for i in df_4.columns], id='tb4')
        ])
    elif tab == 'tab-5':
        return html.Div([
            dash_table.DataTable(df_5.to_dict('records'),[{"name": i, "id": i} for i in df_5.columns], id='tb5')
        ])
    
    elif tab == 'tab-6':
        return html.Div([
            dash_table.DataTable(df_6.to_dict('records'),[{"name": i, "id": i} for i in df_6.columns], id='tb6')
        ])
    elif tab == 'tab-welt':
        return html.Div([
            dash_table.DataTable(df_welt.to_dict('records'),[{"name": i, "id": i} for i in df_welt.columns], id='tb-welt')
        ])

    elif tab == 'tab-7':
        return html.Div([
            dash_table.DataTable(df_7.to_dict('records'),[{"name": i, "id": i} for i in df_7.columns], id='tb7')
        ])
    elif tab == 'tab-14':
        return html.Div([
            dash_table.DataTable(df_14.to_dict('records'),[{"name": i, "id": i} for i in df_14.columns], id='tb14')
        ])
    elif tab == 'tab-15':
        return html.Div([
            dash_table.DataTable(df_15.to_dict('records'),[{"name": i, "id": i} for i in df_15.columns], id='tb15')
        ])

    elif tab == 'tab-deu':
        return html.Div([
            dash_table.DataTable(df_deu.to_dict('records'),[{"name": i, "id": i} for i in df_deu.columns], id='tb_deu')
        ])
      
    elif tab == 'tab-master':
        return html.Div([
            dash_table.DataTable(df_master.to_dict('records'),columns =[{"name": i, "id": i} for i in df_master.columns],id='tb_master',
                                 style_data_conditional = [{
                                     'if': {
                                         'filter_query': '{points} > 19',
                                     },
                                     'backgroundColor': 'green',
                                     'color': 'black'
                                 },
                                     {
                                     'if': {
                                         'filter_query': '{points} <19  && {points} > 13',
                                     },
                                     'backgroundColor': 'yellow',
                                     'color': 'black'
                                 },
                                     {
                                     'if': {
                                         'filter_query': '{points} <13',
                                     },
                                     'backgroundColor': 'red',
                                     'color': 'black'
                                 },
                                     
                                 ])
        ])        
     
    elif tab == 'tab-s':
        return html.Div([
            dash_table.DataTable(df_m1.to_dict('records'),columns =[{"name": i, "id": i} for i in df_m1.columns],id='tb_m1',
                                 style_data_conditional = [{
                                     'if': {
                                         'filter_query': '{points} > 19',
                                     },
                                     'backgroundColor': 'green',
                                     'color': 'black'
                                 },
                                     {
                                     'if': {
                                         'filter_query': '{points} <19  && {points} > 13',
                                     },
                                      'if': {
                                         'filter_query': '{dividend_score} > 9',
                                          'column_id' : 'dividend_score'
                                     },
                                       
                                    
                                         
                                     'backgroundColor': 'yellow',
                                     'color': 'black'
                                 },
                                     
                                    {  
                                         'if': {
                                         'filter_query': '{hgi_score} > 9',
                                          'column_id' : 'hgi_score'
                                     },
                                       
                                    
                                         
                                     'backgroundColor': 'yellow',
                                     'color': 'black'
                                 },
                                     
                                     {  
                                         'if': {
                                         'filter_query': '{levermann_score} > 2',
                                          'column_id' : 'levermann_score'
                                     },
                                       
                                    
                                         
                                     'backgroundColor': 'yellow',
                                     'color': 'black'
                                 },
                                     
                                     
                                     {
                                     'if': {
                                         'filter_query': '{points} <13',
                                     },
                                     'backgroundColor': 'red',
                                     'color': 'black'
                                 },
                                     
                                 ])
        ])        

    
    
if __name__ == '__main__':
    app.run_server()


# In[269]:


# from dash import Dash, Input, Output, callback, dash_table
# import pandas as pd
# import dash_bootstrap_components as dbc

# df = pd.read_csv('https://git.io/Juf1t')

# app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# app.layout = dbc.Container([
#     dbc.Label('Click a cell in the table:'),
#     dash_table.DataTable(df.to_dict('records'),[{"name": i, "id": i} for i in df.columns], id='tbl'),
#     dbc.Alert(id='tbl_out'),
# ])

# @callback(Output('tbl_out', 'children'), Input('tbl', 'active_cell'))
# def update_graphs(active_cell):
#     return str(active_cell) if active_cell else "Click the table"

# if __name__ == "__main__":
#     app.run_server()


# In[ ]:


# import mysql.connector 
# from mysql.connector import Error
# server = "analysis-tool-nonprod.c4rbkrjinf7p.eu-central-1.rds.amazonaws.com" # replace with your server name
# database = "stocks-dev" # replace with your database name
# username = "admin" # replace with your username
# password = "nCk8Ll42KcLU5KdX631L" # replace with your password

# conn = mysql.connector.connect(host = server, user = username,password = password,database = database)

# cursor = conn.cursor()


# delete_stock_table = """
# DROP TABLE stocklist
#  """
# execute_query(conn, delete_stock_table)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[457]:


# database sqlachemy 
#"server=127.0.0.1;uid=root;pwd=12345;database=test"
from sqlalchemy import create_engine
url = "server=localhost;uid=root;pwd=Kingisking123;database=new_schema1"


# In[458]:


my_conn = create_engine(url)


# In[463]:


for row in df_deu.itertuples(): 
    cursor.execute('''INSERT INTO new_schema1.stocks_new4 (stock_id, stockname, dividend, high_growth, leverman,Indexes) Values (?,?,?,?,?,?)''', 
                   row[0],
                   row.name, 
                   row.dividend_score,
                   row.hgi_score,
                   row.levermann_score,
                   row.Index
                  )
    


# In[430]:


for row in df_deu.itertuples():
    a = row[0]
    print(a)


# In[428]:


df_master.head()


# In[ ]:





import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import data
import yfinance as yf
from sentiment_analyzer import calculate_daily_sentiment_scores

nltk.download('stopwords')
nltk.download('wordnet')

def load_datasets():
    # Load each dataset
    company_df = pd.read_csv('Company.csv')
    company_tweet_df = pd.read_csv('Company_Tweet.csv')
    tweets_df = pd.read_csv('company_test.csv')

    # Merge the datasets
    # First, merge Company_Tweet with Tweet based on tweet_id
    merged_df = pd.merge(company_tweet_df, tweets_df, on='tweet_id')

    # Next, merge the result with Company based on ticker_symbol
    final_df = pd.merge(merged_df, company_df, on='ticker_symbol')

    return final_df
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|@\S+|#\S+|[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    text_tokens = text.split()
    text = ' '.join([word for word in text_tokens if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text
def process_time(df):
    # change the Unix or Epoch time format to a normal day time format
    df['post_date'] = pd.to_datetime(df['post_date'], unit='s')
def get_stock_price(company: str):
    ticker = yf.Ticker(company)
    stock = ticker.history(start="2015-01-01", end="2020-12-31")
    stock.reset_index(inplace=True)
    stock['Daily Difference'] = stock['Close'].diff()
    return stock
def process_dataset(final_df, ticker: str):
    company_df = final_df[final_df['ticker_symbol'] == ticker].copy()
    company_df = calculate_daily_sentiment_scores(company_df)
    company_stock = get_stock_price(ticker)
    company_stock['Date'] = company_stock['Date'].dt.date
    company_df = company_df.merge(company_stock, left_on='post_date', right_on='Date', how='left').dropna()
    return company_df




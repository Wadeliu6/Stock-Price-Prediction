import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import data

nltk.download('stopwords')
nltk.download('wordnet')

def load_datasets():
    # Load each dataset
    company_df = pd.read_csv('Company.csv')
    company_tweet_df = pd.read_csv('Company_Tweet.csv')
    tweets_df = pd.read_csv('Tweet.csv')

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




from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
import pandas as pd
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

def aggregate_sentiment(dataframe, group_by_column, sentiment_column):
    aggregated_data = dataframe.groupby(group_by_column)[sentiment_column].mean().reset_index()
    return aggregated_data
# def calculate_z_scores(dataframe, window=15):
#     # Determine the window size: 15 days or the length of the dataset if it's smaller
#     window_size = min(window, len(dataframe))
#
#     # Calculate the rolling mean and standard deviation
#     rolling_mean = dataframe['sentiment'].rolling(window=window_size).mean()
#     rolling_std = dataframe['sentiment'].rolling(window=window_size).std()
#
#     # Calculate Z-scores
#     dataframe['z_score'] = (dataframe['sentiment'] - rolling_mean) / rolling_std
#
#     # Drop the NaN values that result from rolling function
#     dataframe = dataframe.dropna(subset=['z_score'])
#
#     return dataframe


def calculate_daily_sentiment_scores(dataframe, sentiment_column='sentiment', vader_column='vader_score', window=15):
    # Ensure 'post_date' is in datetime format
    dataframe['post_date'] = pd.to_datetime(dataframe['post_date']).dt.date
    # Aggregate daily average sentiment and Vader scores
    daily_sentiment = dataframe.groupby('post_date').agg({sentiment_column: 'mean', vader_column: 'mean'}).reset_index()
    # Determine the window size: 15 days or the length of the dataset if it's smaller
    window_size = min(window, len(daily_sentiment))

    # Calculate rolling mean and standard deviation for sentiment scores
    daily_sentiment['rolling_mean_sentiment'] = daily_sentiment[sentiment_column].rolling(window=window_size, min_periods=1).mean()
    daily_sentiment['rolling_std_sentiment'] = daily_sentiment[sentiment_column].rolling(window=window_size, min_periods=1).std()

    # Calculate Z-scores
    daily_sentiment['z_score'] = (daily_sentiment[sentiment_column] - daily_sentiment['rolling_mean_sentiment']) / daily_sentiment['rolling_std_sentiment']

    # Calculate rolling average for Vader scores
    daily_sentiment['rolling_vader_score'] = daily_sentiment[vader_column].rolling(window=window_size, min_periods=1).mean()

    print(daily_sentiment)
    # Drop NaN values
    daily_sentiment = daily_sentiment.dropna(subset=['z_score', 'rolling_vader_score'])
    return daily_sentiment




def get_vader_score(sentence):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(sentence)['compound']
def vec_fun(data, m, n, max_features=10000, method='tf-idf'):
    if method == 'cv':
        vectorizer = CountVectorizer(ngram_range=(m, n), max_features=max_features)
    elif method == 'tf-idf':
        vectorizer = TfidfVectorizer(ngram_range=(m, n), max_features=max_features)
    else:
        raise ValueError("Invalid method specified. Choose 'cv' or 'tf-idf'.")

    xform_data = vectorizer.fit_transform(data)
    return xform_data, vectorizer.get_feature_names_out()


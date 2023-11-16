from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

def aggregate_sentiment(dataframe, group_by_column, sentiment_column):
    aggregated_data = dataframe.groupby(group_by_column)[sentiment_column].mean().reset_index()
    return aggregated_data
def calculate_z_scores(dataframe, window=15):
    # Calculate the rolling mean and standard deviation
    rolling_mean = dataframe['sentiment'].rolling(window=window).mean()
    rolling_std = dataframe['sentiment'].rolling(window=window).std()

    # Calculate Z-scores
    dataframe['z_score'] = (dataframe['sentiment'] - rolling_mean) / rolling_std

    # Drop the NaN values that result from rolling function
    dataframe = dataframe.dropna(subset=['z_score'])

    return dataframe
def vec_fun(data, m, n, max_features=10000, method='tf-idf'):
    if method == 'cv':
        vectorizer = CountVectorizer(ngram_range=(m, n), max_features=max_features)
    elif method == 'tf-idf':
        vectorizer = TfidfVectorizer(ngram_range=(m, n), max_features=max_features)
    else:
        raise ValueError("Invalid method specified. Choose 'cv' or 'tf-idf'.")

    xform_data = vectorizer.fit_transform(data)
    return xform_data, vectorizer.get_feature_names_out()


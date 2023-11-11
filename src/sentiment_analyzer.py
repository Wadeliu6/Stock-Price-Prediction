from textblob import TextBlob
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

def aggregate_sentiment(dataframe, group_by_column, sentiment_column):
    aggregated_data = dataframe.groupby(group_by_column)[sentiment_column].mean().reset_index()
    return aggregated_data

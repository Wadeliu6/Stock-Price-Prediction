from data_processing import load_datasets, preprocess_text
from sentiment_analyzer import analyze_sentiment, aggregate_sentiment

def main():
    # Load and merge data
    final_df = load_datasets()

    # Preprocess tweets
    final_df['cleaned_tweet'] = final_df['body'].apply(preprocess_text)

    # Perform sentiment analysis
    final_df['sentiment'] = final_df['cleaned_tweet'].apply(analyze_sentiment)

    # Calculate the average sentiment for each company
    aggregated_sentiment = aggregate_sentiment(final_df, 'company_name', 'sentiment')

    print(aggregated_sentiment)

if __name__ == "__main__":
    main()



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sentiment_analyzer import analyze_sentiment, get_vader_score, calculate_daily_sentiment_scores
from data_processing import load_datasets, preprocess_text, get_stock_price, process_time, process_dataset
from Model import linear_regression_model, ranforest_model, lgbm_model

def main():
    # Load and merge data
    final_df = load_datasets()

    # Data Preprocessing
    process_time(final_df)
    final_df['cleaned_tweet'] = final_df['body'].apply(preprocess_text)
    print(final_df.columns)
    final_df['sentiment'] = final_df['cleaned_tweet'].apply(analyze_sentiment)
    print(final_df.columns)
    # final_df = calculate_z_scores(final_df)
    

    # Vectorization with limited features and using sparse matrix
    # vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    # X = vectorizer.fit_transform(final_df['cleaned_tweet'])
    #
    # # Create labels based on the sentiments
    # final_df['label'] = final_df['z_score'].apply(lambda x: 1 if x > 0.5 else -1)
    # y = final_df['label']
    # print(final_df)

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # Initialize and train the RandomForest classifier
    # clf = RandomForestClassifier(random_state=42)
    # clf.fit(X_train, y_train)
    #
    # # Predict and evaluate the classifier
    # y_pred = clf.predict(X_test)
    # print(classification_report(y_test, y_pred))

    # Add vader score
    final_df['vader_score'] = final_df['cleaned_tweet'].apply(get_vader_score)

    # Apple
    # apple_df = final_df[final_df['ticker_symbol'] == 'AAPL'].copy()
    # apple_df = calculate_daily_sentiment_scores(apple_df)
    # apple_stock = get_stock_price('AAPL')
    # apple_stock['Date'] = apple_stock['Date'].dt.date
    # apple_df = apple_df.merge(apple_stock, left_on='post_date', right_on='Date', how='left').dropna()
    apple_df = process_dataset(final_df, 'TSLA')
    #linear_regression_model(apple_df.copy())
    #lgbm_model(apple_df.copy())
    ranforest_model(apple_df.copy())
    
if __name__ == "__main__":
    main()

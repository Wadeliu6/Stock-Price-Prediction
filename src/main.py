import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sentiment_analyzer import analyze_sentiment, calculate_z_scores
from data_processing import load_datasets, preprocess_text, get_stock_price, process_time
from Linear_Regression_Model import linear_regression_model

def main():
    # Load and merge data
    final_df = load_datasets()

    # Data Preprocessing
    process_time(final_df)
    final_df['cleaned_tweet'] = final_df['body'].apply(preprocess_text)
    final_df['sentiment'] = final_df['cleaned_tweet'].apply(analyze_sentiment)
    final_df = calculate_z_scores(final_df)
    

    # Vectorization with limited features and using sparse matrix
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X = vectorizer.fit_transform(final_df['cleaned_tweet'])

    # Create labels based on the sentiments
    final_df['label'] = final_df['z_score'].apply(lambda x: 1 if x > 0.5 else -1)
    y = final_df['label']
    # print(final_df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForest classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate the classifier
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    
    # Apple
    apple_df = final_df[final_df['ticker_symbol'] == 'AAPL']
    apple_df['post_date'] = apple_df['post_date'].dt.date
    apple_stock = get_stock_price('AAPL')
    apple_stock['Date'] = apple_stock['Date'].dt.date
    print(apple_df)
    apple_df = apple_df.merge(apple_stock, left_on='post_date', right_on='Date', how='left').dropna()
    print(apple_df)
    linear_regression_model(apple_df)
    
if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


def linear_regression_model(company_df):
    X = company_df[['z_score', 'vader_score']]
    y = company_df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    visualization(company_df, model, X, 'Linear Regression Model')

    # Regression evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

    print(company_df['Close'])

def ranforest_model(company_df):
    # Random Forest Regressor 
    X = company_df[['z_score', 'vader_score']]
    y = company_df['Close']
    print(company_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)

    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(pd.Series(y_pred))

    visualization(company_df, regressor, X, 'Random Forest Regressor Model')
    
    

# def lgbm_model(company_df):
#     # LightGBM
#     X = company_df[['z_score', 'vader_score']]
#     y = company_df['Close']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Create a lgbm model
#     lgbm = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100)
#
#     lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1')
#
#     y_pred = lgbm.predict(X_test, num_iteration=lgbm.best_iteration_)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     print('RMSE:', rmse)

def gradient_boosting_model(df):
    # Convert index to DateTimeIndex if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Now extract the year, month, and day
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day

    # Define the features and target variable
    X = df[['Year', 'Month', 'Day', 'z_score', 'vader_score']]
    y = df['Close']

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Initialize and train the Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0, loss='squared_error')
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Gradient Boosting Regressor Mean Squared Error:", mse)
    visualization(df, model, X, 'Gradient Boosting Model')

    return model

def visualization(df, model, features, title):
    # draw a plot to show how the predicted value differ from the actual value
    predicted_values = model.predict(features)
    predicted_price = pd.DataFrame({'Date': df['post_date'],'Predicted': predicted_values})

    plt.figure(figsize=(10,6))
    plt.plot(df['post_date'], df['Close'], label='Actual Price')
    plt.plot(predicted_price['Date'], predicted_price['Predicted'], label='Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.xticks(rotation=45)
    plt.title(title)
    plt.legend()
    plt.show()






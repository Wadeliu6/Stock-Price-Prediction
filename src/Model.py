import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


def linear_regression_model(company_df):
    X = company_df[['z_score', 'vader_score']]
    y = company_df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Regression evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

    print(pd.Series(y_pred))

def ranforest_model(company_df):
    # Random Forest Regressor 
    X = company_df[['z_score', 'vader_score']]
    y = company_df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)

    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')

def lgbm_model(company_df):
    # LightGBM
    X = company_df[['z_score', 'vader_score']]
    y = company_df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a lgbm model
    lgbm = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100)

    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1')

    y_pred = lgbm.predict(X_test, num_iteration=lgbm.best_iteration_)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE:', rmse)



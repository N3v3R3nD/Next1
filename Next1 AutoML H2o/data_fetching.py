import logging
from datetime import datetime

import config
import h2o
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler
from config import start_date, end_date, yfinance_symbol

def fetch_and_preprocess_data():
    try:
        # Access the parameters
        forecast_steps = config.forecast_steps
        yfinance_symbol = config.yfinance_symbol

        # Fetch data using yfinance
        logging.info('Fetching data using yfinance')
        data = yf.download(yfinance_symbol, start=start_date, end=end_date)
        logging.info('Data downloaded from Yahoo Finance')
        logging.info('Data shape: %s', data.shape)
        logging.info('First few rows of the data:')

        # Create a shifted version of the 'Open' column as the target
        data['Target'] = data['Open'].shift(-1)

        # Drop the last row, which does not have a target
        data = data[:-1]

        # Fetch macroeconomic data using FRED
        logging.info('Fetching macroeconomic data using FRED')
        gdp = pdr.get_data_fred('GDP', start_date, end_date)
        unemployment = pdr.get_data_fred('UNRATE', start_date, end_date)

        # Preprocess macroeconomic data
        logging.info('Preprocessing macroeconomic data')
        gdp = gdp.resample('D').ffill()  # Fill missing values by forward filling
        unemployment = unemployment.resample('D').ffill()  # Fill missing values by forward filling

        # Merge macroeconomic data with existing data
        logging.info('Merging macroeconomic data with existing data')
        data = pd.merge(data, gdp, how='left', left_index=True, right_index=True)
        data = pd.merge(data, unemployment, how='left', left_index=True, right_index=True)

        # Engineer features from macroeconomic data
        logging.info('Engineering features from macroeconomic data')
        data['GDP Change'] = data['GDP'].pct_change()
        data['Unemployment Change'] = data['UNRATE'].pct_change()

        # Drop original macroeconomic columns
        data = data.drop(columns=['GDP', 'UNRATE'])

        # Add day of the week feature
        data['DayOfWeek'] = data.index.dayofweek # type: ignore

        # Add month of the year feature
        data['Month'] = data.index.month # type: ignore

        # Add is holiday feature
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=data.index.min(), end=data.index.max())
        data['IsHoliday'] = data.index.isin(holidays).astype(int)

        # Filter out non-business days
        data = data[data.index.dayofweek < 5] # type: ignore

        # Add technical indicators
        data['SMA'] = data['Close'].rolling(window=14).mean()
        data['EMA'] = data['Close'].ewm(span=14).mean()
        delta = data['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        data['RSI'] = 100 - (100 / (1 + rs))
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        data['MACD'] = macd - signal
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['20dSTD'] = data['Close'].rolling(window=20).std()
        data['Upper'] = data['MA20'] + (data['20dSTD'] * 2)
        data['Lower'] = data['MA20'] - (data['20dSTD'] * 2)
        data['Cum_Daily_Returns'] = (data['Close'] / data['Close'].shift(1)) - 1
        data['Cumulative_Returns'] = (1 + data['Cum_Daily_Returns']).cumprod()
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        data = data.dropna()

        # Select features
        logging.info('Selecting features')
        features = data[['High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'EMA', 'RSI', 'MACD', 'Upper', 'Lower', 'Cumulative_Returns', 'VWAP', 'GDP Change', 'Unemployment Change', 'Target']]    
        num_features = len(features.columns)  # Get the number of features
        logging.info('Features: ' + str(features.columns.tolist()))  # Log the order of features
        print(features.columns)

        # Handle outliers
        logging.info('Handling outliers')
        Q1 = features.quantile(0.25)
        Q3 = features.quantile(0.75)
        IQR = Q3 - Q1
        features = features[~((features < (Q1 - 1.5 * IQR)) | (features > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Preprocess data
        logging.info('Preprocessing data')
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        # Split data into training and test sets
        logging.info('Splitting data into training and test sets')
        split = int(0.8 * len(features))
        train_features = features.drop('Target', axis=1)[:split]
        test_features = features.drop('Target', axis=1)[split:]
        train_target = features[['Target']][:split]
        test_target = features[['Target']][split:]

        # Fit the scaler on the training data and transform both training and test data
        logging.info('Scaling data')
        feature_scaler = StandardScaler().fit(train_features)
        target_scaler = StandardScaler().fit(train_target)

        scaled_train_features = feature_scaler.transform(train_features)
        scaled_test_features = feature_scaler.transform(test_features)
        scaled_train_target = target_scaler.transform(train_target)
        scaled_test_target = target_scaler.transform(test_target)

        # Create the dataset for training
        logging.info('Creating dataset for training')
        X_train, Y_train = [], []
        forecast_steps = forecast_steps 
        for i in range(forecast_steps, len(train_features)):
            X_train.append(scaled_train_features[i-forecast_steps:i, :])
            Y_train.append(scaled_train_target[i, 0])

        # Create the dataset for testing
        logging.info('Creating dataset for testing')
        X_test, Y_test = [], []
        for i in range(forecast_steps, len(test_features)):
            X_test.append(scaled_test_features[i-forecast_steps:i, :]) # type: ignore
            Y_test.append(scaled_test_target[i, 0])  # type: ignore

        # Convert lists to numpy arrays
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_test, Y_test = np.array(X_test), np.array(Y_test)

        # Reshape the data to 2D
        X_train_2d = X_train.reshape((X_train.shape[0], -1))
        X_test_2d = X_test.reshape((X_test.shape[0], -1))

        # Convert data to H2O data frames
        X_train_h2o = h2o.H2OFrame(X_train_2d)
        Y_train_h2o = h2o.H2OFrame(Y_train)
        X_test_h2o = h2o.H2OFrame(X_test_2d)
        Y_test_h2o = h2o.H2OFrame(Y_test)

        # Get the original column names (exclude the target)
        original_column_names = train_features.columns.tolist()

        # Create new column names for the reshaped data
        reshaped_column_names = [f"{name}_{i}" for name in original_column_names for i in range(forecast_steps)]

        # Set the column names in the H2O data frames
        X_train_h2o.columns = reshaped_column_names
        X_test_h2o.columns = reshaped_column_names

        return X_train, Y_train, X_test, Y_test, train_features, test_features, data, scaled_train_target, scaled_test_target, forecast_steps, target_scaler, num_features
    except Exception as e:
        logging.error(f"Error during data fetching and preprocessing: {e}")
        raise

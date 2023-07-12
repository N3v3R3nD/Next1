# data_fetching.py
import os
import yfinance as yf
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import pandas_datareader as pdr
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import StandardScaler
import config
def fetch_and_preprocess_data():

    # Access the parameters
    look_back = config.look_back
    yfinance_symbol = config.yfinance_symbol

    # Set up logging
    logging.basicConfig(filename='next1.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Fetch data using yfinance
    logging.info('Fetching data using yfinance')
    today = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(yfinance_symbol, start='1993-01-29', end=today)
    logging.info('Data downloaded from Yahoo Finance')
    logging.info(f'Data shape: {data.shape}')
    logging.info('First few rows of the data:')

    # Fetch macroeconomic data using FRED
    logging.info('Fetching macroeconomic data using FRED')
    start_date = '1993-01-29'
    end_date = today
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
    data['DayOfWeek'] = data.index.dayofweek

    # Add month of the year feature
    data['Month'] = data.index.month

    # Add is holiday feature
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=data.index.min(), end=data.index.max())
    data['IsHoliday'] = data.index.isin(holidays).astype(int)

    # Filter out non-business days
    data = data[data.index.dayofweek < 5]

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
    features = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SMA', 'EMA', 'RSI', 'MACD', 'Upper', 'Lower', 'Cumulative_Returns', 'VWAP', 'GDP Change', 'Unemployment Change']]
    num_features = len(features.columns)  # Get the number of features
    logging.info('Features: ' + str(features.columns.tolist()))  # Log the order of features

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
    train_features = features[:split]
    test_features = features[split:]

    # Fit the scaler on the training data and transform both training and test data
    logging.info('Scaling data')
    scaled_train_features = feature_scaler.fit_transform(train_features)
    scaled_test_features = feature_scaler.transform(test_features)

    # Do the same for the target variable
    train_target = features[['Open']][:split]
    test_target = features[['Open']][split:]
    scaled_train_target = target_scaler.fit_transform(train_target)
    scaled_test_target = target_scaler.transform(test_target)

    # Create the dataset for training
    logging.info('Creating dataset for training')
    X_train, Y_train = [], []
    look_back = look_back 
    for i in range(look_back, len(train_features)):
        X_train.append(scaled_train_features[i-look_back:i, :])
        Y_train.append(scaled_train_target[i, 0])

    # Create the dataset for testing
    logging.info('Creating dataset for testing')
    X_test, Y_test = [], []
    for i in range(look_back, len(test_features)):
        X_test.append(scaled_test_features[i-look_back:i, :])
        Y_test.append(scaled_test_target[i, 0]) 

    # Convert lists to numpy arrays
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    return X_train, Y_train, X_test, Y_test, train_features, test_features, data, scaled_train_target, scaled_test_target, look_back, target_scaler, num_features

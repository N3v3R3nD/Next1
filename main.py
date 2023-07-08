# main.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import yfinance as yf
import psycopg2
import numpy as np
import pandas as pd
import logging
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from tensorflow.keras import regularizers
from scikeras.wrappers import KerasRegressor
from datetime import datetime, timedelta
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas_datareader as pdr
from sklearn.model_selection import KFold
from pandas.tseries.holiday import USFederalHolidayCalendar
import data_fetching
import db_operations
import model_evaluation
from hyperparameter_tuning import create_model, hyperparameter_tuning
from model_training import train_model  # Import the train_model function

# Load the configuration
with open('config.json') as f:
    config = json.load(f)

# Access the parameters
use_kfold = config['use_kfold']
kfold_splits = config['kfold_splits']
early_stopping_patience = config['early_stopping_patience']
look_back = config['look_back']
model_params = config['model_params']

# Set up logging
logging.basicConfig(filename='next1.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set the format for console output
console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console_handler.setFormatter(console_formatter)

# Add the console handler to the logger
logging.getLogger('').addHandler(console_handler)

logging.info('Starting script')

# Fetch and preprocess data
logging.info('Fetching and preprocessing data')
X_train, Y_train, X_test, Y_test, train_features, test_features, data, scaled_train_target, scaled_test_target, look_back, target_scaler, num_features = data_fetching.fetch_and_preprocess_data()

logging.info('Hyperparameter tuning')
# Hyperparameter tuning
best_params = hyperparameter_tuning(X_train, Y_train, look_back, train_features.shape[1], train_features, use_bayesian_optimization=True)
model_params = best_params.copy()

# Reshape data for LSTM
logging.info('Reshaping data for LSTM')
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], train_features.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], test_features.shape[1]))

# Get the minimum and maximum 'Close' prices
min_close_price = data['Close'].min()
max_close_price = data['Close'].max()

# Manually compute the scaled value for the first 'Close' price
first_close_price = data['Close'].iloc[0]
scaled_first_close_price = (first_close_price - min_close_price) / (max_close_price - min_close_price)

# Check if the manually computed scaled value matches the first value in scaled_train_target
logging.info("Manually computed scaled value for the first 'Close' price: " + str(scaled_first_close_price))
logging.info("First value in scaled_train_target: " + str(scaled_train_target[0][0]))

# Train the model
model, history = train_model(X_train, Y_train, X_test, Y_test, look_back, num_features, model_params)

# Generate predictions
logging.info('Generating predictions')
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Access the underlying Keras model
keras_model = model.model_

# Save the model
model.model_.save('trained_model.h5')
logging.info('Saved trained model to disk')

# Evaluate model
train_predict, test_predict, train_rmse, test_rmse, train_mae, test_mae = model_evaluation.evaluate_model(Y_train, Y_test, train_predict, test_predict, target_scaler)

# Connect to the database
conn, cur = db_operations.connect_to_db()

# Create tables
db_operations.create_tables(cur)

# Insert data
db_operations.insert_data(cur, history, Y_train, train_predict, test_predict, target_scaler)
db_operations.insert_evaluation_results(cur, train_rmse, test_rmse, train_mae, test_mae)

# Close the connection
db_operations.close_connection(conn)

logging.info('Script completed')

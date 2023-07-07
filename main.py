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
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import pandas_datareader as pdr
from sklearn.model_selection import KFold
from pandas.tseries.holiday import USFederalHolidayCalendar
import data_fetching
import db_operations
import model_evaluation
from hyperparameter_tuning import create_model, hyperparameter_tuning

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
best_params = hyperparameter_tuning(X_train, Y_train, look_back, train_features.shape[1], train_features)
model_params = best_params.copy()
# Print the first few elements of Y_train and Y_test

# Create model with best parameters

model = KerasRegressor(model=create_model, look_back=look_back, num_features=train_features.shape[1], **model_params, verbose=0)

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

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Define 5-fold cross validation
use_kfold = False  # Set this flag to True to enable KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42) if use_kfold else None

# Log the KFold configuration
logging.info(f'KFold cross-validation: {use_kfold}')

# Define a list to store the model objects of each fold
models = []

# Loop over each fold
if kf:
    for train_index, val_index in kf.split(X_train):
        # Create training and validation sets for this fold
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]

        # Create a new model for this fold
        model = KerasRegressor(model=create_model, look_back=look_back, **model_params, verbose=0)
        
        # Fit the model and store the model object
        history = model.fit(X_train_fold, Y_train_fold, validation_data=(X_val_fold, Y_val_fold), verbose=1, callbacks=[early_stopping])

        # Add the model object to the list
        models.append(model)

else:
    # No KFold cross-validation, train a single model on the whole dataset
    # Wrap Keras model with KerasRegressor
    # Create model with best parameters
    model = KerasRegressor(model=create_model, look_back=look_back, num_features=num_features, **model_params, verbose=0)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])


    models.append(model)

# Access the underlying Keras model and its history
keras_model = model.model_
history = keras_model.history.history

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

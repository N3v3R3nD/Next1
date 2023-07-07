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
X_train, Y_train, X_test, Y_test, train_features, test_features, data, scaled_train_target, scaled_test_target, look_back, target_scaler = data_fetching.fetch_and_preprocess_data()

# Print the first few elements of Y_train and Y_test
logging.info("First few elements of Y_train: " + str(Y_train[:5]))
logging.info("First few elements of Y_test: " + str(Y_test[:5]))

# Print the last few elements of Y_train and Y_test
logging.info("Last few elements of Y_train: " + str(Y_train[-5:]))
logging.info("Last few elements of Y_test: " + str(Y_test[-5:]))

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


# Function to create LSTM model
def create_model(units=100, optimizer='adam', dropout_rate=0.0):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(look_back, train_features.shape[1]), kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    model.add(Dropout(dropout_rate))  # Add dropout layer
    model.add(LSTM(units=units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    model.add(Dropout(dropout_rate))  # Add dropout layer
    model.add(LSTM(units=units, return_sequences=False, kernel_regularizer=tf.keras.regularizers.L2(0.01)))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Wrap Keras model with KerasRegressor
model_params = {'units': 100, 'optimizer': 'adam', 'dropout_rate': 0.0}
model = KerasRegressor(model=create_model, **model_params, verbose=0)

# Define hyperparameters for RandomizedSearchCV
param_dist = {
    'units': [50, 100, 150, 200],  # More units can help model complexity
    'batch_size': [16, 32, 64, 128],  # Larger batch sizes can speed up training
    'epochs': [50, 100, 150],  # More epochs can lead to more stable models
    'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # Dropout can help prevent overfitting
    'optimizer': ['Adam']  # Different optimizers can have different effects on training
}

# UseTimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=10)  # More splits can provide a more robust estimate of model performance

# Define random_search before the try block
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=tscv, n_iter=20, n_jobs=-1)  # More iterations can explore the hyperparameter space more thoroughly

# Check if hyperparameters file exists
try:
    with open('best_params.json', 'r') as f:
        best_params = json.load(f)
    logging.info(f'Loaded parameters: {best_params}')
except FileNotFoundError:
    # Perform RandomizedSearchCV
    logging.info('Performing RandomizedSearchCV')
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=tscv, verbose=2, n_iter=10, n_jobs=-1)  # Use all cores

    start_time = time.time()
    random_search.fit(X_train, Y_train)
    elapsed_time = time.time() - start_time
    logging.info(f'RandomizedSearchCV completed. Elapsed time: {elapsed_time} seconds')

    # Get the best parameters
    best_params = random_search.best_params_
    logging.info(f'Best parameters: {best_params}')

    # Save the best parameters
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)
    logging.info('Saved best parameters')

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
        model = KerasRegressor(model=create_model, verbose=0, **best_params)

        # Fit the model and store the model object
        history = model.fit(X_train_fold, Y_train_fold, validation_data=(X_val_fold, Y_val_fold), verbose=1, callbacks=[early_stopping])

        # Add the model object to the list
        models.append(model)

else:
    # No KFold cross-validation, train a single model on the whole dataset
    model = KerasRegressor(model=create_model, verbose=0, **best_params)
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])

    models.append(model)

logging.info('Training completed')

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
train_predict, test_predict = model_evaluation.evaluate_model(Y_train, Y_test, train_predict, test_predict, target_scaler)
# Connect to the database
conn, cur = db_operations.connect_to_db()

# Create tables
db_operations.create_tables(cur)

# Insert data
db_operations.insert_data(cur, history, Y_train, train_predict, test_predict, target_scaler)

# Close the connection
db_operations.close_connection(conn)

logging.info('Script completed')

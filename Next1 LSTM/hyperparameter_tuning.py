# hyperparameter_tuning.py
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import regularizers
from skopt import BayesSearchCV
import tensorflow as tf
import time
import logging
import config
import json


# Extract hyperparameters from config
param_dist = config.param_dist
units = param_dist['units']
batch_size = param_dist['batch_size']
epochs = param_dist['epochs']
dropout_rate = param_dist['dropout_rate']
optimizer = param_dist['optimizer']
tscv_splits = config.tscv_splits
tuner = config.tuner

def create_model(look_back, num_features, units=100, optimizer='adam', dropout_rate=0.0):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(look_back, num_features), kernel_regularizer=regularizers.L2(0.01)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=True, kernel_regularizer=regularizers.L2(0.01)))
    model.add(Dropout(dropout_rate)) 
    model.add(LSTM(units=units, return_sequences=False, kernel_regularizer=regularizers.L2(0.01)))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def hyperparameter_tuning(X_train, Y_train, look_back, feature_num, train_features):
    # Wrap Keras model with KerasRegressor
    model_params = {
    'units': units,
    'optimizer': optimizer,
    'dropout_rate': dropout_rate,
    'num_features': feature_num
        }
    model = KerasRegressor(model=create_model, look_back=look_back, **model_params, verbose=0)
    
    # Define hyperparameters for RandomizedSearchCV or BayesSearchCV
    param_dist = {
        'units': units,
        'batch_size': batch_size,
        'epochs': epochs,
        'dropout_rate': dropout_rate,
        'optimizer': optimizer
    }

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=config.tscv_splits) # More splits can provide a more robust estimate of model performance

    # Define search_cv before the try block
    if tuner == 'bayesian':
        logging.info('Using Bayesian Optimization for hyperparameter tuning')
        search_cv = BayesSearchCV(
            estimator=model,
            search_spaces=param_dist,
            n_iter=50,
            cv=tscv,
            n_jobs=-1
        )
    elif tuner == 'grid':
        logging.info('Using Grid Search for hyperparameter tuning')
        search_cv = GridSearchCV(
            estimator=model,
            param_grid=param_dist,
            cv=tscv,
            n_jobs=-1
        )
    elif tuner == 'random':
        logging.info('Using Random Search for hyperparameter tuning')
        search_cv = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            n_jobs=-1
        )
    else:
        logging.error(f'Invalid tuner: {tuner}. Please set tuner to "bayesian", "grid", or "random" in the configuration file.')
        return
    # Check if hyperparameters file exists
    try:
        with open('best_params.json', 'r') as f:
            best_params = json.load(f)
        logging.info(f'Loaded parameters: {best_params}')
    except FileNotFoundError:
        # Perform RandomizedSearchCV or BayesSearchCV
        logging.info('Performing hyperparameter search')
        
        start_time = time.time()
        search_cv.fit(X_train, Y_train)
        elapsed_time = time.time() - start_time
        logging.info(f'Hyperparameter search completed. Elapsed time: {elapsed_time} seconds')

        # Get the best parameters
        best_params = search_cv.best_params_
        logging.info(f'Best parameters: {best_params}')

        # Save the best parameters
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f)
        logging.info('Saved best parameters')

    return best_params

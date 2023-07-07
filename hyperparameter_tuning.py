# hyperparameter_tuning.py
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import time
import json
import logging


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

def hyperparameter_tuning(X_train, Y_train, look_back, feature_num):
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

    return best_params

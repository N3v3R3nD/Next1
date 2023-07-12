import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from h2o.exceptions import H2OResponseError
import logging

def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def calculate_mae(actual, predicted):
    return mean_absolute_error(actual, predicted)

def calculate_rae(actual, predicted):
    return np.sum(np.abs(predicted - actual)) / np.sum(np.abs(actual - np.mean(actual)))

def calculate_rse(actual, predicted):
    return np.sum(np.square(predicted - actual)) / np.sum(np.square(actual - np.mean(actual)))

def calculate_r2(actual, predicted):
    sse = np.sum(np.square(predicted - actual))
    sst = np.sum(np.square(actual - np.mean(actual)))
    return 1 - (sse / sst)

def evaluate_model(model, actual_train, actual_test, predicted_train, predicted_test):
    try:
        train_rmse = calculate_rmse(actual_train, predicted_train)
        test_rmse = calculate_rmse(actual_test, predicted_test)
        train_mae = calculate_mae(actual_train, predicted_train)
        test_mae = calculate_mae(actual_test, predicted_test)
        train_rae = calculate_rae(actual_train, predicted_train)
        test_rae = calculate_rae(actual_test, predicted_test)
        train_rse = calculate_rse(actual_train, predicted_train)
        test_rse = calculate_rse(actual_test, predicted_test)
        train_r2 = calculate_r2(actual_train, predicted_train)
        test_r2 = calculate_r2(actual_test, predicted_test)
        
        logging.info(f'Train RMSE: {train_rmse}')
        logging.info(f'Test RMSE: {test_rmse}')
        logging.info(f'Train MAE: {train_mae}')
        logging.info(f'Test MAE: {test_mae}')
        logging.info(f'Train RAE: {train_rae}')
        logging.info(f'Test RAE: {test_rae}')
        logging.info(f'Train RSE: {train_rse}')
        logging.info(f'Test RSE: {test_rse}')
        logging.info(f'Train R2: {train_r2}')
        logging.info(f'Test R2: {test_r2}')

        return train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2
    except H2OResponseError as e:
        logging.error(f'Error: {e}')

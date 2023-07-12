# model_evaluation.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

def evaluate_model(Y_train, Y_test, train_predict, test_predict, target_scaler):
    # Reshape predictions into 2D arrays
    train_predict = train_predict.reshape(-1, 1)
    test_predict = test_predict.reshape(-1, 1)

    # Invert predictions using target_scaler
    train_predict = target_scaler.inverse_transform(train_predict)
    Y_train = target_scaler.inverse_transform(Y_train.reshape(-1, 1))
    test_predict = target_scaler.inverse_transform(test_predict)
    Y_test = target_scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Calculate root mean squared error
    train_rmse = np.sqrt(mean_squared_error(Y_train, train_predict))
    logging.info(f'Train Score: {train_rmse} RMSE')
    test_rmse = np.sqrt(mean_squared_error(Y_test, test_predict))
    logging.info(f'Test Score: {test_rmse} RMSE')

    # Calculate mean absolute error
    train_mae = mean_absolute_error(Y_train, train_predict)
    logging.info(f'Train Score: {train_mae} MAE')
    test_mae = mean_absolute_error(Y_test, test_predict)
    logging.info(f'Test Score: {test_mae} MAE')

    return train_predict, test_predict, train_rmse, test_rmse, train_mae, test_mae

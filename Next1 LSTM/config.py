# config.py

# Number of previous time steps to use as input features
look_back = 21

# Ticker symbol for the stock to predict
yfinance_symbol = "SPY"

# Whether to use K-Fold cross-validation during model training
use_kfold = False

# Number of folds to use if use_kfold is True
kfold_splits = 2

# Number of epochs with no improvement after which training will be stopped
early_stopping_patience = 2

# Parameters for the model
model_params = {
    # Number of units in the LSTM layer
    "units": 100,
    # Batch size for training
    "batch_size": 32,
    # Number of epochs for training
    "epochs": 100,
    # Dropout rate
    "dropout_rate": 0.2,
    # Optimizer for training
    "optimizer": "adam"
}

# Parameters for hyperparameter tuning
param_dist = {
    # Possible values for the number of units in the LSTM layer
    "units": [50, 100, 150, 200],
    # Possible values for the batch size
    "batch_size": [16, 32, 64, 128],
    # Possible values for the number of epochs
    "epochs": [50, 100, 150],
    # Possible values for the dropout rate
    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    # Possible values for the optimizer
    "optimizer": ["Adam"]
}

# Details for connecting to the database
database = {
    # Host name
    "host": "localhost",
    # Database name
    "database": "stock",
    # Username
    "user": "postgres",
    # Password
    "password": "test123"
}

# Number of splits for time series cross-validation
tscv_splits = 10

# Method for hyperparameter tuning ("grid" for grid search, "random" for random search, "bayesian" for Bayesian optimization)
tuner = "grid"

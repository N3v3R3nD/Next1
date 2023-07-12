# config.py
from datetime import datetime

start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
# end_date = '2023-06-23'
# Number of previous time steps to use as input features
forecast_steps = 21

target_column_name = 'Open'

# Ticker symbol for the stock to predict
yfinance_symbol = "ES=F"

# Number of epochs with no improvement after which training will be stopped
early_stopping_patience = 5


# Details for connecting to the database
database = {
    # Host name
    "host": "snuffleupagus.db.elephantsql.com",
    # Database name
    "database": "rzpjtxcf",
    # Username
    "user": "rzpjtxcf",
    # Password
    "password": "lbFXUWGzaOw_aju7fmq0mNkt39T3fAKf"
}

# AutoML settings
automl_settings = {
#   'max_runtime_secs': 3600,              # Maximum time for AutoML to run (in seconds)
    'max_models': 40,                    # Maximum number of models to build
    'seed': 1,                             # Random seed for reproducibility
    'balance_classes': False,
    'include_algos': ['DRF', 'GBM', 'XGBoost', 'DeepLearning'],  # Algorithms to include in AutoML
    'keep_cross_validation_models': True,   # Whether to keep cross-validated models in the AutoML leaderboard
    'keep_cross_validation_predictions': True,  # Whether to keep cross-validated predictions in the AutoML leaderboard
    'verbosity': 'info'                     # Set the verbosity level of the AutoML process
}

# Other configuration options...

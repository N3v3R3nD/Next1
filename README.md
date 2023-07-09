# Next1: Stock Market Forecasting Application

Next1 is a sophisticated Python application designed for forecasting stock market prices. It leverages machine learning techniques to predict future prices based on historical data and various technical indicators.

## Table of Contents

- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

### Data Fetching and Preprocessing

The application fetches historical stock price data and macroeconomic data using the `yfinance` and `pandas_datareader` libraries. This is handled by the `fetch_and_preprocess_data` function in `data_fetching.py`. The fetched data is then preprocessed and cleaned, with special consideration given to outliers and missing values. This includes resampling, forward filling, and outlier handling using the IQR method.

### Feature Engineering

The application engineers several features from the fetched data, including technical indicators like SMA, EMA, RSI, MACD, and Bollinger Bands. It also takes into account macroeconomic features like GDP Change and Unemployment Change.

### Model Training and Evaluation

The application trains a machine learning model on the processed data. The best model parameters are stored in `best_params.json`. The model's performance is evaluated using metrics such as RMSE and MAE. This functionality is encapsulated in the `evaluate_model` function in `model_evaluation.py`.

### Hyperparameter Tuning

The application includes a hyperparameter tuning function that employs either Random Search or Bayesian Optimization to identify the optimal parameters for the model. This functionality is implemented in the `hyperparameter_tuning` function in `hyperparameter_tuning.py`.

### Database Operations

The forecast data, loss data, and actual vs. predicted prices are stored in a PostgreSQL database for long-term storage and further analysis. The functions for these operations are housed in `db_operations.py`.

## File Structure

- `main.py`: The main script that orchestrates the data fetching, preprocessing, model training, evaluation, and database operations.
- `data_fetching.py`: Contains the `fetch_and_preprocess_data` function that fetches, preprocesses, and splits the data into training and test sets.
- `db_operations.py`: Contains functions for connecting to the database, creating tables, inserting data, and closing the connection.
- `model_evaluation.py`: Contains the `evaluate_model` function that evaluates the model's performance using RMSE and MAE.
- `hyperparameter_tuning.py`: Contains the `hyperparameter_tuning` function that performs hyperparameter tuning using either Random Search or Bayesian Optimization.
- `best_params.json`: A JSON file that stores the optimal parameters for the model.

## Installation

To use this application, follow the steps below:

1. **Install Python and PostgreSQL**: The application requires Python and PostgreSQL. Make sure both are installed on your machine.
2. **Clone the repository**: Clone the Next1v2 repository to your local machine using the following command in your terminal: `git clone https://github.com/N3v3R3nD/Next1.git`.
3. **Navigate to the project directory**: Change your current directory to the Next1v2 directory: `cd Next1v2`.
4. **Install the required Python libraries**: Install the required Python libraries by executing `pip install -r requirements.txt`. The required libraries include `numpy`, `pandas`, `sklearn`, `psycopg2`, `yfinance`, `pandas_datareader`, and `keras`.

## Usage

This application is designed to be easy to use. Follow the steps below to get started:

1. **Prepare the Database**: Make sure your PostgreSQL database is up and running. Update the database connection details in `db_operations.py`.
2. **Run the application**: Run the application using the command: `python main.py`.

The application will fetch the data, preprocess it, train the model, evaluate it, and store the results in the PostgreSQL database.

## Contributing

Contributions are welcome! Please feel free to submit a pull request. If you have any questions or issues, feel free to open an issue in the repository.

## License

All rights reserved. This project and the associated source code is the property of the repository owner. No part of this software, including this file, may be copied, modified, propagated, or distributed except with the prior written permission of the owner.

PLEASE BE AWARE that this software is provided 'as is', without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.


# Next1: Stock Market Forecasting Application

Next1v2 is a sophisticated Python application designed for stock market forecasting. It leverages machine learning techniques to predict future prices based on historical data and various technical indicators.

## Table of Contents

- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

### Data Fetching and Preprocessing

The application fetches historical stock price data and macroeconomic data using `yfinance` and `pandas_datareader`. This is done in the `fetch_and_preprocess_data` function in `data_fetching.py`. The fetched data is preprocessed and cleaned, handling outliers and missing values. This includes resampling, forward filling, and outlier handling using IQR.

### Feature Engineering

The application engineers features from the data, including technical indicators like SMA, EMA, RSI, MACD, and Bollinger Bands. It also includes macroeconomic features like GDP Change and Unemployment Change.

### Model Training and Evaluation

The application trains a machine learning model on the processed data. The model parameters are stored in `best_params.json`. The model is evaluated using metrics like RMSE and MAE. This is done in the `evaluate_model` function in `model_evaluation.py`.

### Hyperparameter Tuning

The application includes a hyperparameter tuning function that can use either Random Search or Bayesian Optimization to find the best parameters for the model. This is done in the `hyperparameter_tuning` function in `hyperparameter_tuning.py`.

### Database Operations

The forecast data, loss data, and actual vs. predicted prices are stored in a PostgreSQL database. The functions for these operations are in `db_operations.py`.

## File Structure

- `main.py`: The main script that orchestrates the data fetching, preprocessing, model training, evaluation, and database operations.
- `data_fetching.py`: Contains the `fetch_and_preprocess_data` function that fetches, preprocesses, and splits the data into training and test sets.
- `db_operations.py`: Contains functions for connecting to the database, creating tables, inserting data, and closing the connection.
- `model_evaluation.py`: Contains the `evaluate_model` function that evaluates the model using RMSE and MAE.
- `hyperparameter_tuning.py`: Contains the `hyperparameter_tuning` function that performs hyperparameter tuning using either Random Search or Bayesian Optimization.
- `best_params.json`: A JSON file that stores the best parameters for the model.

## Installation

To use this application, you need to have Python and PostgreSQL installed on your machine. You also need to install the required Python libraries, which include `numpy`, `pandas`, `sklearn`, `psycopg2`, `yfinance`, and `pandas_datareader`.

## Usage

This application is designed to be easy to use. Follow the steps below to get started:

1. **Clone the repository**: Clone the Next1v2 repository to your local machine using the following command in your terminal:
    `
    git clone https://github.com/N3v3R3nD/Next1v2.git
    `

2. **Navigate to the project directory**: Change your current directory to the Next1v2 directory:

    `
    cd Next1v2
   `

3. **Install the required Python libraries**: Before running the application, you need to install the required Python libraries. You can do this using pip:

    `
    pip install -r requirements.txt
    `

4. **Run the application**: Now, you can run the application using the following command:

    `
    python main.py
    `

The application will fetch the data, preprocess it, train the model, evaluate it, and store the results in a PostgreSQL database.


## Contributing


Contributions are welcome! Please feel free to submit a pull request. If you have any questions or issues, feel free to open an issue in the repository.

## License

All rights reserved. This project and the associated source code is the property of the repository owner. No part of this software, including this file, may be copied, modified, propagated, or distributed except with the prior written permission of the owner.

PLEASE BE AWARE that this software is provided 'as is', without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.



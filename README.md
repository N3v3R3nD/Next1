# Next1v2: Stock Market Forecasting Application

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

### Database Operations

The forecast data, loss data, and actual vs. predicted prices are stored in a PostgreSQL database. The functions for these operations are in `db_operations.py`.

## File Structure

- `main.py`: The main script that orchestrates the data fetching, preprocessing, model training, evaluation, and database operations.
- `data_fetching.py`: Contains the `fetch_and_preprocess_data` function that fetches, preprocesses, and splits the data into training and test sets.
- `db_operations.py`: Contains functions for connecting to the database, creating tables, inserting data, and closing the connection.
- `model_evaluation.py`: Contains the `evaluate_model` function that evaluates the model using RMSE and MAE.
- `best_params.json`: A JSON file that stores the best parameters for the model.

## Installation

To use this application, you need to have Python and PostgreSQL installed on your machine. You also need to install the required Python libraries, which include `numpy`, `pandas`, `sklearn`, `psycopg2`, `yfinance`, and `pandas_datareader`.

## Usage

Once you have the prerequisites, you can clone the repository and run `main.py`:

\```bash
git clone https://github.com/N3v3R3nD/Next1v2.git
cd Next1v2
python main.py
\```

## Contributing

\```markdown
Contributions are welcome! Please feel free to submit a pull request. If you have any questions or issues, feel free to open an issue in the repository.
\```

## License

This project is licensed under the terms of the MIT license. This allows you to use, modify, and distribute the code in this repository for your own projects as long as you include the original copyright notice and disclaimers.


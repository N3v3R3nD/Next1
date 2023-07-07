# Next1v2

Next1v2 is a Python application designed for stock market forecasting. It uses machine learning techniques to predict future prices based on historical data and various technical indicators.

## Features

- **Data Fetching**: The application fetches historical stock price data and macroeconomic data using `yfinance` and `pandas_datareader`. This is done in the `fetch_and_preprocess_data` function in `data_fetching.py`.

- **Data Preprocessing**: The fetched data is preprocessed and cleaned, handling outliers and missing values. This includes resampling, forward filling, and outlier handling using IQR.

- **Feature Engineering**: The application engineers features from the data, including technical indicators like SMA, EMA, RSI, MACD, and Bollinger Bands. It also includes macroeconomic features like GDP Change and Unemployment Change.

- **Model Training**: The application trains a machine learning model on the processed data. The model parameters are stored in `best_params.json`.

- **Model Evaluation**: The model is evaluated using metrics like RMSE and MAE. This is done in the `evaluate_model` function in `model_evaluation.py`.

- **Database Operations**: The forecast data, loss data, and actual vs. predicted prices are stored in a PostgreSQL database. The functions for these operations are in `db_operations.py`.

## Usage

To use this application, you need to have Python and PostgreSQL installed on your machine. You also need to install the required Python libraries, which include `numpy`, `pandas`, `sklearn`, `psycopg2`, `yfinance`, and `pandas_datareader`.

Once you have the prerequisites, you can clone the repository and run `main.py`:

```bash
git clone https://github.com/N3v3R3nD/Next1v2.git
cd Next1v2
python main.py
Contributing

Contributions are welcome! Please feel free to submit a pull request.
License

This project is licensed under the terms of the MIT license.

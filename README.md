Next1v2

Next1v2 is a Python application designed for stock market forecasting. It uses machine learning techniques to predict future prices based on historical data and various technical indicators.
Features

    Fetches historical stock price data and macroeconomic data.
    Preprocesses and cleans the data, handling outliers and missing values.
    Engineers features from the data, including technical indicators like SMA, EMA, RSI, MACD, and Bollinger Bands.
    Trains a machine learning model on the processed data.
    Evaluates the model using metrics like RMSE and MAE.
    Stores the forecast data, loss data, and actual vs. predicted prices in a PostgreSQL database.

Files

    main.py: The main script that orchestrates the data fetching, preprocessing, model training, evaluation, and database operations.
    data_fetching.py: Contains the fetch_and_preprocess_data function that fetches, preprocesses, and splits the data into training and test sets.
    db_operations.py: Contains functions for connecting to the database, creating tables, inserting data, and closing the connection.
    model_evaluation.py: Contains the evaluate_model function that evaluates the model using RMSE and MAE.
    best_params.json: A JSON file that stores the best parameters for the model.

Usage

To use this application, you need to have Python and PostgreSQL installed on your machine. You also need to install the required Python libraries, which include numpy, pandas, sklearn, psycopg2, yfinance, and pandas_datareader.

Once you have the prerequisites, you can clone the repository and run main.py:

bash

git clone https://github.com/N3v3R3nD/Next1v2.git
cd Next1v2
python main.py

Contributing

Contributions are welcome! Please feel free to submit a pull request.
License

This project is licensed under the terms of the MIT license.

Please review and modify this draft as needed.

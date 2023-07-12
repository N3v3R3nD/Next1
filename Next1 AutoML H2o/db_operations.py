import logging
from datetime import datetime, timedelta

import psycopg2

import config
import numpy as np
import json

# Extract database credentials from config
db_config = config.database
host = db_config['host']
database = db_config['database']
user = db_config['user']
password = db_config['password']

def connect_to_db():
    try:
        # Connect to the database
        logging.info('Connecting to the database')
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        cur = conn.cursor()
        return conn, cur
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        raise

def create_tables(cur):
    try:
        # Create actual_vs_predicted table if it doesn't exist
        logging.info('Creating actual_vs_predicted table if it does not exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS actual_vs_predicted (
                execution_id SERIAL,
                date DATE,
                actual_price FLOAT,
                predicted_price FLOAT,
                PRIMARY KEY (execution_id, date)
            )
        """)

        # Create evaluation_results table if it doesn't exist
        logging.info('Creating evaluation_results table if it doesn\'t exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                execution_id SERIAL PRIMARY KEY,
                train_rmse FLOAT,
                test_rmse FLOAT,
                train_mae FLOAT,
                test_mae FLOAT,
                train_rae FLOAT,
                test_rae FLOAT,
                train_rse FLOAT,
                test_rse FLOAT,
                train_r2 FLOAT,
                test_r2 FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create fetched_data table if it doesn't exist
        logging.info('Creating fetched_data table if it does not exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fetched_data (
                execution_id SERIAL,
                date DATE,
                open_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                close_price FLOAT,
                adj_close_price FLOAT,
                volume FLOAT,
                PRIMARY KEY (execution_id, date)
            )
        """)

        # Create forecasted_prices table if it doesn't exist
        logging.info('Creating forecasted_prices table if it doesn\'t exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forecasted_prices (
                execution_id SERIAL,
                date DATE,
                forecasted_price FLOAT,
                PRIMARY KEY (execution_id, date)
            )
        """)

        # Create execution_settings table if it doesn't exist
        logging.info('Creating execution_settings table if it doesn\'t exist')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS execution_settings (
                execution_id SERIAL PRIMARY KEY,
                config_settings TEXT,
                model_parameters TEXT
            )
        """)
    except Exception as e:
        logging.error(f"Error creating tables: {e}")
        raise

def get_next_execution_id(cur):
    try:
        # Create fetched_data table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fetched_data (
                execution_id SERIAL,
                date DATE,
                open_price FLOAT,
                high_price FLOAT,
                low_price FLOAT,
                close_price FLOAT,
                adj_close_price FLOAT,
                volume FLOAT,
                PRIMARY KEY (execution_id, date)
            )
        """)

        # Fetch the last execution_id from the fetched_data table
        cur.execute("SELECT MAX(execution_id) FROM fetched_data")
        last_execution_id = cur.fetchone()[0]

        # If this is the first execution, start from 1, otherwise increment the last execution_id by 1
        execution_id = 1 if last_execution_id is None else last_execution_id + 1

        return execution_id
    except Exception as e:
        logging.error(f"Error fetching next execution id: {e}")
        raise


def insert_execution_settings(cur, execution_id, config, model):
    try:
        logging.info('Inserting execution settings into the database')

        # Create a new dictionary with only the necessary settings
        config_dict = {
            'forecast_steps': config.forecast_steps,
            'database': config.database,
            # Add any other settings you need
        }

        # Convert config dictionary and model parameters to string
        config_settings = json.dumps(config_dict)
        model_parameters = json.dumps(model.params)

        # SQL query to insert execution settings into the database
        query = """
            INSERT INTO execution_settings (execution_id, config_settings, model_parameters) 
            VALUES (%s, %s, %s)
        """
        cur.execute(query, (execution_id, config_settings, model_parameters))
    except Exception as e:
        logging.error(f"Error inserting execution settings: {e}")
        raise

def insert_data(cur, execution_id, Y_train, train_preds, forecast, target_scaler):
    try:
        # Insert actual and predicted prices into the database
        logging.info('Inserting actual and predicted prices into the database')

        if Y_train.shape[0] != train_preds.shape[0]:
            raise ValueError("Length of Y_train and train_preds don't match")

        for i, actual_price in enumerate(Y_train):
            date = (datetime.today() - timedelta(days=len(Y_train) - i - 1)).strftime('%Y-%m-%d')  # Calculate the correct date
            actual_price = target_scaler.inverse_transform(actual_price.reshape(-1, 1))[0][0]
            predicted_price = target_scaler.inverse_transform(train_preds[i].reshape(-1, 1))[0][0]

            cur.execute(f"""
                INSERT INTO actual_vs_predicted (execution_id, date, actual_price, predicted_price) 
                VALUES ({execution_id}, '{date}', {actual_price}, {predicted_price}) 
                ON CONFLICT (execution_id, date) DO UPDATE 
                SET actual_price = {actual_price}, predicted_price = {predicted_price}
            """)
    except Exception as e:
        logging.error(f"Error inserting data: {e}")
        raise

def insert_forecast(cur, execution_id, forecast, target_scaler):
    try:
        logging.info("Inserting forecast")
        # Convert forecast to list if it's a numpy array
        if isinstance(forecast, np.ndarray):
            forecast = forecast.tolist()

        # Get today's date
        today = datetime.today()

        # SQL query to insert or update forecast in the database
        query = """
            INSERT INTO forecasted_prices (execution_id, date, forecasted_price) 
            VALUES (%s, %s, %s) 
            ON CONFLICT (execution_id, date) 
            DO UPDATE SET forecasted_price = EXCLUDED.forecasted_price
        """
        # Insert or update each forecasted price in the database
        for i, price in enumerate(forecast):
            date = today + timedelta(days=i+1)  # The date is the current date plus the forecast horizon

            # Inverse transform the forecasted price
            forecasted_price = target_scaler.inverse_transform(np.array([[price]]))[0][0]

            cur.execute(query, (execution_id, date, forecasted_price)) # added execution_id here
    except Exception as e:
        logging.error(f"Error inserting forecast: {e}")
        raise

def insert_fetched_data(cur, execution_id, data):
    try:
        logging.info('Inserting fetched data into the database')
        for i in range(len(data)):
            date = data.index[i].strftime('%Y-%m-%d')  # Get the date as a string
            open_price = data['Open'].iloc[i]  # Get the open price
            high_price = data['High'].iloc[i]  # Get the high price
            low_price = data['Low'].iloc[i]  # Get the low price
            close_price = data['Close'].iloc[i]  # Get the close price
            adj_close_price = data['Adj Close'].iloc[i]  # Get the adjusted close price
            volume = data['Volume'].iloc[i]  # Get the volume

            cur.execute(f"""
                INSERT INTO fetched_data (execution_id, date, open_price, high_price, low_price, close_price, adj_close_price, volume) 
                VALUES ({execution_id}, '{date}', {open_price}, {high_price}, {low_price}, {close_price}, {adj_close_price}, {volume}) 
                ON CONFLICT (execution_id, date) DO UPDATE 
                SET open_price = {open_price}, high_price = {high_price}, low_price = {low_price}, close_price = {close_price}, adj_close_price = {adj_close_price}, volume = {volume}
            """)
    except Exception as e:
        logging.error(f"Error inserting fetched data: {e}")
        raise

def insert_evaluation_results(cur, execution_id, train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2):
    try:
        logging.info("Inserting evaluation results")
        # SQL query to insert evaluation results into the database
        query = """
            INSERT INTO evaluation_results (execution_id, train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (execution_id, train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2))
        logging.debug(f"Inserted evaluation results")
    except Exception as e:
        logging.error(f"Error inserting evaluation results: {e}")
        raise

def close_connection(conn):
    try:
        logging.info("Closing connection")
        # Close the database connection
        conn.close()
    except Exception as e:
        logging.error(f"Error closing connection: {e}")
        raise
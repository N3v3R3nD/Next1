import psycopg2
from datetime import datetime, timedelta
import logging
import json

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Extract database credentials from config
db_config = config['database']
db_host = db_config['host']
db_name = db_config['database']
db_user = db_config['user']
db_password = db_config['password']

def connect_to_db():
    # Connect to the database
    logging.info('Connecting to the database')
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password
    )
    cur = conn.cursor()
    return conn, cur

def create_tables(cur):
    # Create forecast_data table if it doesn't exist
    logging.info('Creating forecast_data table if it doesn\'t exist')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecast_data (
            date DATE PRIMARY KEY,
            forecast FLOAT
        )
    """)

    # Create loss_data table if it doesn't exist
    logging.info('Creating loss_data table if it doesn\'t exist')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS loss_data (
            epoch INTEGER PRIMARY KEY,
            loss FLOAT
        )
    """)

    # Create actual_vs_predicted table if it doesn't exist
    logging.info('Creating actual_vs_predicted table if it does not exist')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS actual_vs_predicted (
            date DATE PRIMARY KEY,
            actual_price FLOAT,
            predicted_price FLOAT
        )
    """)

    # Create evaluation_results table if it doesn't exist
    logging.info('Creating evaluation_results table if it doesn\'t exist')
    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id SERIAL PRIMARY KEY,
            train_rmse FLOAT,
            test_rmse FLOAT,
            train_mae FLOAT,
            test_mae FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

def insert_data(cur, history, Y_train, train_predict, test_predict, target_scaler):    # Insert forecast data into the database
    logging.info('Inserting forecast data into the database')
    for i in range(len(test_predict)):
        date = (datetime.today() + timedelta(days=i)).strftime('%Y-%m-%d')
        forecast = test_predict[i][0]
        cur.execute(f"INSERT INTO forecast_data (date, forecast) VALUES ('{date}', {forecast}) ON CONFLICT (date) DO UPDATE SET forecast = {forecast}")

    # Insert loss data into the database
    logging.info('Inserting loss data into the database')
    for i, loss in enumerate(history['loss']):
        cur.execute(f"INSERT INTO loss_data (epoch, loss) VALUES ({i}, {loss}) ON CONFLICT (epoch) DO UPDATE SET loss = {loss}")

    # Insert actual and predicted prices into the database
    logging.info('Inserting actual and predicted prices into the database')

    if len(Y_train) != len(train_predict):
        raise ValueError("Length of Y_train and train_predict don't match")

    for i in range(len(Y_train)):
        date = (datetime.today() - timedelta(days=len(Y_train) - i - 1)).strftime('%Y-%m-%d')  # Calculate the correct date
        actual_price = target_scaler.inverse_transform(Y_train[i].reshape(-1, 1))[0][0]
        predicted_price = train_predict[i][0]
        
        # Print the values for comparison
        # print(f"Actual Price: {actual_price}, Predicted Price: {predicted_price}")
        
        cur.execute(f"""
            INSERT INTO actual_vs_predicted (date, actual_price, predicted_price) 
            VALUES ('{date}', {actual_price}, {predicted_price}) 
            ON CONFLICT (date) DO UPDATE 
            SET actual_price = {actual_price}, predicted_price = {predicted_price}
        """)
        

def insert_evaluation_results(cur, train_rmse, test_rmse, train_mae, test_mae):
    # Insert evaluation results into the database
    logging.info('Inserting evaluation results into the database')
    cur.execute(f"""
        INSERT INTO evaluation_results (train_rmse, test_rmse, train_mae, test_mae) 
        VALUES ({train_rmse}, {test_rmse}, {train_mae}, {test_mae})
    """)
def close_connection(conn):
    # Commit changes and close connection
    conn.commit()
    conn.close()

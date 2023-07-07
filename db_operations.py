import psycopg2
from datetime import datetime, timedelta
import logging

def connect_to_db():
    # Connect to the database
    logging.info('Connecting to the database')
    conn = psycopg2.connect(
        host='localhost',
        database='stock',
        user='postgres',
        password='test123'
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


def insert_data(cur, history, Y_train, train_predict, test_predict):
    # Insert forecast data into the database
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
    for i in range(len(Y_train)):
        date = (datetime.today() - timedelta(days=len(Y_train)-i))
        date_str = date.strftime('%Y-%m-%d')
        actual_price = Y_train[i]
        predicted_price = train_predict[i][0]
        cur.execute(f"""
            INSERT INTO actual_vs_predicted (date, actual_price, predicted_price) 
            VALUES ('{date_str}', {actual_price}, {predicted_price}) 
            ON CONFLICT (date) DO UPDATE 
            SET actual_price = {actual_price}, predicted_price = {predicted_price}
        """)

def close_connection(conn):
    # Commit changes and close connection
    conn.commit()
    conn.close()

import logging
import os

import config
import data_fetching
import h2o
import model_evaluation
import numpy as np
from model_training import train_model
import psycopg2
import db_operations

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())

forecast_steps = config.forecast_steps

# Set up logging
logging.basicConfig(filename='next1.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set the format for console output
console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console_handler.setFormatter(console_formatter)

# Add the console handler to the logger
logging.getLogger('').addHandler(console_handler)

# Initialize the H2O cluster
h2o.init()

# Log to file as well
logging.info('Starting script')

try:
    # Connect to the database
    conn, cur = db_operations.connect_to_db()

    # Get the next execution_id
    execution_id = db_operations.get_next_execution_id(cur)

    # Create tables
    db_operations.create_tables(cur)

    # Fetch and preprocess data
    logging.info('Fetching and preprocessing data')
    X_train, Y_train, X_test, Y_test, train_features, test_features, data, scaled_train_target, scaled_test_target, forecast_steps, target_scaler, num_features = data_fetching.fetch_and_preprocess_data()

    # Insert fetched data into the database
    db_operations.insert_fetched_data(cur, execution_id, data)

    # Commit changes
    conn.commit()

    # Check that the shapes of the input data are as expected
    assert X_train.shape[1] == forecast_steps, 'Unexpected shape of X_train'
    assert X_test.shape[1] == forecast_steps, 'Unexpected shape of X_test'
    assert Y_train.ndim == 1, 'Unexpected shape of Y_train'
    assert Y_test.ndim == 1, 'Unexpected shape of Y_test'

    # Call the train_model function and get the results
    model_path, model, (train_preds, test_preds), forecast = train_model(X_train, Y_train, X_test, Y_test, forecast_steps, num_features, model_params=None)

    # Log shapes for debugging
    logging.info('Shape of Y_train: %s', np.shape(Y_train))
    logging.info('Shape of train_preds: %s', np.shape(train_preds))
    logging.info('Shape of test_preds: %s', np.shape(test_preds))
    logging.info('Shape of forecast: %s', np.shape(forecast))
    
    # Evaluate model
    train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2 = model_evaluation.evaluate_model(model, Y_train, Y_test, train_preds, test_preds)

    # Connect to the database
    conn, cur = db_operations.connect_to_db()
    
    # Insert execution settings
    db_operations.insert_execution_settings(cur, execution_id, config, model)

    # Insert data
    db_operations.insert_data(cur, execution_id, Y_train, train_preds, forecast, target_scaler)

    # Insert forecast into the database
    db_operations.insert_forecast(cur, execution_id, forecast, target_scaler)
    
    # Insert evaluation results
    db_operations.insert_evaluation_results(cur, execution_id, train_rmse, test_rmse, train_mae, test_mae, train_rae, test_rae, train_rse, test_rse, train_r2, test_r2)

    # Commit changes
    conn.commit()

except ValueError as ve:
    logging.error('ValueError occurred: %s', ve)
except IOError as ioe:
    logging.error('IOError occurred: %s', ioe)
except Exception as e:
    logging.error('An error occurred: %s', e)
    # Optionally, you can raise the exception again to stop the script
    raise
finally:
    # Close connection
    db_operations.close_connection(conn)  # type: ignore
    h2o.cluster().shutdown()

logging.info('Script completed')

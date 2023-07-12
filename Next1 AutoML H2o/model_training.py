# model_training.py
import logging 
import h2o
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
import config
import os

forecast_steps = config.forecast_steps
automl_settings = config.automl_settings

def train_model(X_train, Y_train, X_test, Y_test, forecast_steps, num_features, model_params):
    try:
        logging.info("Starting model training")
        
        # Check if the model exists and load it
        model_filename = "MyModel"
        model_file_path = os.path.join(".", model_filename)
        if os.path.exists(model_file_path):
            model = h2o.load_model(model_file_path)
        else:
            
            # Reshape data if it's a 3D numpy array
            if isinstance(X_train, np.ndarray) and len(X_train.shape) == 3:
                X_train = X_train.reshape(X_train.shape[0], -1)
            if isinstance(X_test, np.ndarray) and len(X_test.shape) == 3:
                X_test = X_test.reshape(X_test.shape[0], -1)

            # Convert data to H2O data frames
            logging.info(f"Convert data to H2O data frames")
            X_train_h2o = h2o.H2OFrame(X_train if isinstance(X_train, pd.DataFrame) else X_train.tolist())
            Y_train_h2o = h2o.H2OFrame(Y_train if isinstance(Y_train, pd.DataFrame) else Y_train.tolist())
            X_test_h2o = h2o.H2OFrame(X_test if isinstance(X_test, pd.DataFrame) else X_test.tolist())
            Y_test_h2o = h2o.H2OFrame(Y_test if isinstance(Y_test, pd.DataFrame) else Y_test.tolist())

            # Set the column names to match the original data frames
            X_train_h2o.columns = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else [f'C{i+1}' for i in range(X_train.shape[1])]
            Y_train_h2o.columns = [config.target_column_name]
            X_test_h2o.columns = X_test.columns.tolist() if isinstance(X_test, pd.DataFrame) else [f'C{i+1}' for i in range(X_test.shape[1])]
            Y_test_h2o.columns = [config.target_column_name]
            # Combine features and target into a single data frame
            train_data = X_train_h2o.cbind(Y_train_h2o)
            test_data = X_test_h2o.cbind(Y_test_h2o)

            y = config.target_column_name  # Define the target column name
            x = train_data.columns  # Define the feature column names

            # Check if the target column name exists in the feature set
            if y in x:
                x.remove(y)
            else:
                logging.error(f'Target column name {y} not found in feature set. Please check the column names.')

            logging.debug(f"Target column name: {y}")
            logging.debug(f"List of column names: {x}")

            # Run AutoML
            aml = H2OAutoML(**config.automl_settings)
            aml.train(x=x, y=y, training_frame=train_data)

            # Get the best model
            model = aml.leader

            # Save the model
            model_path = h2o.save_model(model=model, path=".", force=True)
            
            # Make predictions
            train_preds = model.predict(train_data)
            test_preds = model.predict(test_data)

            # Convert predictions to numpy array and flatten them
            train_preds = train_preds.as_data_frame().values.flatten()
            test_preds = test_preds.as_data_frame().values.flatten()

            # Log the shapes for debugging
            logging.debug(f'Shape of train_preds: {train_preds.shape}')
            logging.debug(f'Shape of Y_train: {Y_train.shape}')
            logging.debug(f'Shape of test_preds: {test_preds.shape}')
            logging.debug(f'Shape of Y_test: {Y_test.shape}')

            # Check that the shapes of the predictions are as expected
            assert train_preds.shape == Y_train.shape, 'Unexpected shape of train_preds'
            assert test_preds.shape == Y_test.shape, 'Unexpected shape of test_preds'

            # Print or log the predictions
            logging.info(f'Train predictions: {train_preds}')
            logging.info(f'Test predictions: {test_preds}')

            # Generate new input data for forecast
            logging.debug(f'X_test shape: {X_test.shape}')  
            logging.debug(f'forecast_steps: {forecast_steps}')  
            forecast_input = X_test[-forecast_steps:]  # Get the most recent observations

            # Flatten the forecast_input before converting to H2OFrame
            forecast_input_flattened = forecast_input.reshape(-1, forecast_input.shape[-1])

            # Convert forecast input to H2O data frame
            forecast_input_h2o = h2o.H2OFrame(forecast_input_flattened)

            # Make forecast
            forecast = model.predict(forecast_input_h2o)

            # Convert forecast to numpy array
            forecast = forecast.as_data_frame().values.flatten()
            logging.info("Model training completed")
            return model_path, model, (train_preds, test_preds), forecast
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

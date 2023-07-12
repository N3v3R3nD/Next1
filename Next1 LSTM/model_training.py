from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold
from hyperparameter_tuning import create_model  # Import the create_model function
import logging  # Import the logging module
import config


# Access the parameters
use_kfold = config.use_kfold
kfold_splits = config.kfold_splits
early_stopping_patience = config.early_stopping_patience

def train_model(X_train, Y_train, X_test, Y_test, look_back, num_features, model_params):
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

    # Define 5-fold cross validation

    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=42) if use_kfold else None

    # Log the KFold configuration
    logging.info(f'KFold cross-validation: {use_kfold}')
    if use_kfold:
        logging.info(f'Starting KFold Cross Validation splits: {kfold_splits}')  # Log the number of splits

    # Define a list to store the model objects of each fold
    models = []

    # Loop over each fold
    if kf:
        for train_index, val_index in kf.split(X_train):
            # Create training and validation sets for this fold
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]

            # Create a new model for this fold
            model = KerasRegressor(model=create_model, look_back=look_back, num_features=num_features, **model_params, verbose=0)

            # Fit the model and store the model object
            logging.info(f'Starting training for fold {len(models) + 1}')
            history = model.fit(X_train_fold, Y_train_fold, validation_data=(X_val_fold, Y_val_fold), verbose=1, callbacks=[early_stopping])
            logging.info(f'Training completed for fold {len(models) + 1}')
            # Add the model object to the list
            models.append(model)

    else:
        # No KFold cross-validation, train a single model on the whole dataset
        # Wrap Keras model with KerasRegressor
        # Create model with best parameters
        model = KerasRegressor(model=create_model, look_back=look_back, num_features=num_features, **model_params, verbose=0)
        logging.info('Starting training on the whole dataset')
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])
        logging.info('Training completed on the whole dataset')
        models.append(model)

    # Access the underlying Keras model and its history
    keras_model = model.model_
    history = keras_model.history.history

    return model, history

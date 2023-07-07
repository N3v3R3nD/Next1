# model_training.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import logging

def create_and_train_model(X_train, Y_train, X_test, Y_test, best_params, use_kfold=False):
    # Function to create LSTM model
    def create_model(units=100, optimizer='adam', dropout_rate=0.0):
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(look_back, train_features.shape[1]), kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        model.add(Dropout(dropout_rate))  # Add dropout layer
        model.add(LSTM(units=units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        model.add(Dropout(dropout_rate))  # Add dropout layer
        model.add(LSTM(units=units, return_sequences=False, kernel_regularizer=tf.keras.regularizers.L2(0.01)))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    # Wrap Keras model with KerasRegressor
    model = KerasRegressor(model=create_model, verbose=0, **best_params)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Train the model
    if use_kfold:
        # TODO: Implement KFold cross-validation
        pass
    else:
        # No KFold cross-validation, train a single model on the whole dataset
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), verbose=1, callbacks=[early_stopping])

    logging.info('Training completed')

    return model, history

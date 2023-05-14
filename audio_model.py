from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import numpy as np 

def build_model(input_shape, num_classes):
    # Build the CNN model
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.5))

    # Build the LSTM model
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_model(X_train, y_train, X_test, y_test, epochs, batch_size):
    # Determine the input shape and number of classes
    input_shape = (X_train.shape[1], 1)
    num_classes = y_train.shape[1]

    model = build_model(input_shape, num_classes)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Convert the predictions to labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Calculate the Concordance Correlation Coefficient (CCC)
    def ccc(y_true, y_pred):
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        covar = np.cov(y_true, y_pred)[0][1]

        rho = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred)**2)
        return rho

    ccc_value = ccc(y_test, y_pred_labels)

    return mse, ccc_value

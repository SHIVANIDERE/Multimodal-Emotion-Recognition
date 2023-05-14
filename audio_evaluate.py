import numpy as np
from sklearn.metrics import mean_squared_error
from audio_features import extract_features
from keras.models import load_model

def evaluate_model(model_file, X_test, y_test):
    # Load the trained model
    model = load_model(model_file)

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

    print('MSE: {:.2f}'.format(mse))
    print('CCC: {:.2f}'.format(ccc_value))

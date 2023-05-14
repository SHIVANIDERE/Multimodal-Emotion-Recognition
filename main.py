import audio_features
import audio_model
import audio_evaluate
from sklearn import train_test_split
import numpy as np
from keras.utils import to_categorical

def predict_emotion(audio_file):
    # Load the model
    model = audio_model.load_model('audio_model.h5')

    # Extract features from the audio file
    features = audio_features.extract_features_from_audio(audio_file)

    # Reshape the features for the model
    features = features.reshape(1, features.shape[0], 1)

    # Make a prediction using the model
    prediction = model.predict(features)

    # Get the predicted label
    labels = ["Anger","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
    predicted_label = labels[np.argmax(prediction)]

    return predicted_label


if __name__ == '__main__':
    # Load the data
    data = audio_features.load_data('extracted_features.csv')

    # Extract the features
    X, y = audio_features.extract_features(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the labels to one-hot encoded vectors
    num_classes = len(np.unique(y))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Train the model
    model = audio_model.train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)

    # Save the model
    model.save('audio_model.h5')

    # Evaluate the model
    audio_evaluate.evaluate_model('audio_model.h5', X_test, y_test)

    # Make a prediction using the model
    audio_file = 'utterance_6.wav'
    predicted_label = predict_emotion(audio_file)
    print(f"Predicted emotion: {predicted_label}")

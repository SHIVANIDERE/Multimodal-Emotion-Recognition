import numpy as np
from keras.models import load_model
import librosa

def audio_prediction(audio_path, model_path):
    # Load the model
    model = load_model(model_path)

    # Load audio file
    audio, sr = librosa.load(audio_path, sr=None)

    # Extract features
    pitch = librosa.yin(audio, fmin=50, fmax=500)
    volume = librosa.feature.rms(y = audio)
    mfcc = librosa.feature.mfcc(y = audio, sr=sr, n_mfcc=13)
    energy = librosa.feature.rms(y = audio, frame_length=2048, hop_length=512)
    chroma = librosa.feature.chroma_stft(y = audio, sr=sr, hop_length=512)

    # Concatenate features into one vector
    features = np.concatenate([pitch.reshape(1,-1), volume, mfcc, energy, chroma], axis=1)

    # Reshape feature vector for input to model
    features = features.reshape(1, features.shape[0], features.shape[1])

    # Predict emotion label
    emotions = ["Anger","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
    y_pred = model.predict(features)
    emotion_label = emotions[np.argmax(y_pred)]

    return emotion_label

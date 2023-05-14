import numpy as np
import pandas as pd
import librosa

def load_data(filename):
    # Load the data from the CSV file
    data = pd.read_csv(filename)

    return data

def extract_features(data):
    # Initialize the lists for the features and labels
    X = []
    y = []

    # Iterate over the rows of the DataFrame
    for index, row in data.iterrows():
        # Load the audio file
        audio_file = row['Final Project/Audio Model/audio_files/']
        y, sr = librosa.load(audio_file, sr=None)

        # Extract the pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches)
        pitch_std = np.std(pitches)

        # Extract the volume features
        volume = np.mean(np.abs(y))
        volume_db = librosa.amplitude_to_db(volume)

        # Extract the MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Extract the energy features
        energy = np.sum(np.square(y))
        energy_db = librosa.power_to_db(energy)

        # Extract the tempo features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_mean = np.mean(tempo)

        # Combine the features into a single feature vector
        features = np.concatenate((pitches.flatten(), [pitch_mean, pitch_std], [volume, volume_db], mfcc_mean, mfcc_std, [energy, energy_db], [tempo_mean]))

        # Append the feature vector and label to the lists
        X.append(features)
        y.append(row['label'])

    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sounddevice as sd
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load Audio and Extract Features
def load_audio_features(file_path, sr=22050, n_mfcc=20):
    y, _ = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T

# Prepare Dataset
def load_dataset(dataset_path):
    genres = os.listdir(dataset_path)
    X, Y = [], []
    genre_dict = {genre: idx for idx, genre in enumerate(genres)}
    
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        for file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file)
            features = load_audio_features(file_path)
            X.append(features)
            Y.append(genre_dict[genre])
    
    return np.array(X), np.array(Y), genre_dict

# Build AI Model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')  # Assuming 10 genres
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Model
def train_model(model, X_train, Y_train, X_val, Y_val, epochs=20):
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=32)

# Streamlit Web Interface
def web_interface(model, genre_dict):
    st.title("Music Genre Classification")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        
        features = load_audio_features("temp_audio.wav")
        features = np.expand_dims(features, axis=0)
        
        prediction = model.predict(features)
        genre_index = np.argmax(prediction)
        genre_name = list(genre_dict.keys())[list(genre_dict.values()).index(genre_index)]
        
        st.write(f"Predicted Genre: {genre_name}")

# Example Usage
if __name__ == "__main__":
    dataset_path = "music_genre_dataset"
    X, Y, genre_dict = load_dataset(dataset_path)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)
    train_model(model, X_train, Y_train, X_val, Y_val, epochs=10)
    
    web_interface(model, genre_dict)
    
    print("Web-based Music Genre Classification Ready!")

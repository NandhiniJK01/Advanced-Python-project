import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import sounddevice as sd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout

# Load and Preprocess Audio Data
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def extract_features(audio, frame_length=1024, hop_length=512):
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    return np.abs(stft)

# Build AI Model for Noise Reduction
def build_model():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 1)),
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=True),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Training Placeholder
def train_model(model, noisy_audio, clean_audio, epochs=10):
    model.fit(noisy_audio, clean_audio, epochs=epochs, batch_size=32)

# Apply AI-based Noise Cancellation
def enhance_speech(model, noisy_features):
    enhanced_audio = model.predict(noisy_features)
    return librosa.istft(enhanced_audio)

# Real-time Noise Cancellation
def real_time_noise_cancellation(model, duration=5, sr=16000):
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete. Processing...")
    
    noisy_features = extract_features(audio.flatten())
    enhanced_audio = enhance_speech(model, noisy_features)
    
    print("Playing enhanced speech...")
    sd.play(enhanced_audio, samplerate=sr)
    sd.wait()

# Example Usage
if __name__ == "__main__":
    noisy_audio = load_audio("noisy_speech.wav")
    clean_audio = load_audio("clean_speech.wav")
    
    noisy_features = extract_features(noisy_audio)
    clean_features = extract_features(clean_audio)
    
    model = build_model()
    train_model(model, noisy_features, clean_features, epochs=5)
    
    real_time_noise_cancellation(model)
    
    print("Real-time Speech Enhancement Complete!")

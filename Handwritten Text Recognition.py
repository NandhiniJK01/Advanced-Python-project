import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Bidirectional, Input, Reshape, BatchNormalization, Dropout
import matplotlib.pyplot as plt
import os

# Load Dataset (Assuming IAM Handwriting Database or Custom Dataset)
def load_dataset(dataset_path):
    images, labels = [], []
    for file in os.listdir(dataset_path):
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(dataset_path, file), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 64)) / 255.0  # Increased size for better recognition
            images.append(img)
            labels.append(file.split(".")[0])  # Assuming filename contains the label
    return np.array(images), labels

# Define CNN + LSTM Model for Multiline Handwritten Text Recognition
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = Reshape(target_shape=(64, 256*128))(x)  # Reshape for LSTM
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train Model
def train_model(model, X_train, Y_train, epochs=15):
    model.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_split=0.2)

# Real-time Multiline Handwritten Text Recognition (Using Webcam)
def real_time_recognition(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 64)) / 255.0  # Resized for better accuracy
        gray = np.expand_dims(gray, axis=(0, -1))
        
        prediction = model.predict(gray)
        predicted_text = np.argmax(prediction)
        
        cv2.putText(frame, f"Prediction: {predicted_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Handwritten Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    dataset_path = "handwritten_dataset"
    X, Y = load_dataset(dataset_path)
    X = X.reshape(-1, 64, 256, 1)
    
    num_classes = len(set(Y))
    model = build_model((64, 256, 1), num_classes)
    train_model(model, X, Y, epochs=15)
    
    real_time_recognition(model)
    print("Multiline Handwritten Text Recognition Ready!")

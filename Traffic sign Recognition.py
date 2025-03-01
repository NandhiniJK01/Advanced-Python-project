import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import geopy.distance
from tensorflow.keras.models import load_model

# Load the pre-trained traffic sign recognition model
model = load_model('traffic_sign_model.h5')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Predefined GPS locations for speed warning (example coordinates)
speed_limit_zones = {
    "Speed Limit 20": (37.7749, -122.4194),
    "Speed Limit 30": (37.7750, -122.4195)
}

# Function to get current GPS location (mocked for now)
def get_current_location():
    return (37.7749, -122.4194)  # Replace with actual GPS module integration

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define class labels (replace with actual sign names)
class_labels = ['Speed Limit 20', 'Speed Limit 30', 'Stop', 'Yield', 'No Entry', 'Traffic Light', 'Roundabout', 'Pedestrian Crossing']

# Function to preprocess input images
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input
    return image

# Function to detect and classify traffic signs
def detect_traffic_signs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signs = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 50, param1=50, param2=30, minRadius=10, maxRadius=100)
    
    if signs is not None:
        signs = np.uint16(np.around(signs))
        for i in signs[0, :]:
            x, y, r = i[0], i[1], i[2]
            roi = frame[y-r:y+r, x-r:x+r]  # Extract region of interest
            if roi.size == 0:
                continue
            preprocessed_roi = preprocess_image(roi)
            prediction = model.predict(preprocessed_roi)
            label = class_labels[np.argmax(prediction)]
            
            # Draw circle around detected sign and label it
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, label, (x-30, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Speak the detected sign
            speak(f"Detected {label}")
            
            # Check GPS location for speed warning
            if label in speed_limit_zones:
                current_location = get_current_location()
                distance = geopy.distance.geodesic(current_location, speed_limit_zones[label]).meters
                if distance < 100:  # Alert if within 100 meters
                    speak(f"Approaching {label} zone. Please adjust speed.")
    
    return frame

# Real-time traffic sign recognition
def real_time_sign_recognition():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_traffic_signs(frame)
        cv2.imshow("Traffic Sign Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_sign_recognition()
    print("Traffic Sign Recognition System Ready!")

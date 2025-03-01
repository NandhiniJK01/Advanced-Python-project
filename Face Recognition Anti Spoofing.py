import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained anti-spoofing model
anti_spoofing_model = load_model('anti_spoofing_model.h5')

# Load face recognition model (pre-trained)
face_recognition_model = load_model('face_recognition_model.h5')

# Load mask detection model
mask_detection_model = load_model('mask_detection_model.h5')

# Function to detect and recognize faces
def detect_and_recognize_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (64, 64)) / 255.0
        face_roi_expanded = np.expand_dims(face_roi_resized, axis=0)
        
        # Anti-spoofing detection
        spoof_prediction = anti_spoofing_model.predict(face_roi_expanded)
        is_real = spoof_prediction[0][0] > 0.5
        
        # Mask detection
        mask_prediction = mask_detection_model.predict(face_roi_expanded)
        mask_label = "Mask Detected" if mask_prediction[0][0] > 0.5 else "No Mask"
        
        if is_real:
            recognition_prediction = face_recognition_model.predict(face_roi_expanded)
            identity = np.argmax(recognition_prediction)
            label = f"Person {identity}"  # Replace with actual name mapping
        else:
            label = "Spoof Attempt Detected!"
            
        color = (0, 255, 0) if is_real else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, mask_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    return frame

# Real-time face recognition & anti-spoofing detection
def real_time_face_recognition():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_and_recognize_faces(frame)
        cv2.imshow("Face Recognition, Mask Detection & Anti-Spoofing", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_face_recognition()
    print("Face Recognition, Mask Detection & Anti-Spoofing System Ready!")

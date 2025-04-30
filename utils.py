import cv2
import numpy as np
import time
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    """
    Calculate the eye aspect ratio (EAR)
    A lower EAR indicates a closed eye
    """
    # Compute the euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def detect_faces(frame, face_cascade):
    """
    Detect faces in a frame using OpenCV's cascade classifier
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces, gray

def preprocess_face_for_model(face_roi, target_size=(145, 145)):
    """
    Preprocess the face region for model prediction
    """
    # Resize the face to the model's expected input size
    face_roi = cv2.resize(face_roi, target_size)
    
    # Normalize pixel values
    preprocessed_face = face_roi / 255.0
    
    # Add batch dimension
    preprocessed_face = np.expand_dims(preprocessed_face, axis=0)
    
    return preprocessed_face

def draw_prediction(frame, faces, predictions, alarm_status, alarm_start_time=None):
    """
    Draw bounding boxes and prediction labels on the frame
    """
    for (x, y, w, h), prediction in zip(faces, predictions):
        color = (0, 255, 0) if prediction < 0.5 else (0, 0, 255)  # Green for active, Red for drowsy
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        status = "Active" if prediction < 0.5 else "Drowsy"
        confidence = 1 - prediction if prediction < 0.5 else prediction
        
        # Display prediction text
        text = f"{status}: {confidence:.2f}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add alarm text if active
        if alarm_status:
            elapsed_time = time.time() - alarm_start_time
            cv2.putText(frame, f"DROWSINESS ALERT! ({elapsed_time:.1f}s)", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return frame

def check_drowsiness(prediction, consecutive_frames, threshold=0.7):
    """
    Check if the driver is drowsy based on consecutive predictions
    """
    if prediction >= threshold:
        consecutive_frames += 1
    else:
        consecutive_frames = 0
        
    return consecutive_frames

def initialize_video_writer(frame, output_path):
    """
    Initialize a video writer for saving the output video
    """
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h), True)
    return writer
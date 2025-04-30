import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import time
import os
import tempfile
import pygame
from PIL import Image
import pandas as pd
import h5py
import sys

# Set page configuration
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide"
)

# Initialize session states
if 'alarm_status' not in st.session_state:
    st.session_state['alarm_status'] = False
if 'alarm_start_time' not in st.session_state:
    st.session_state['alarm_start_time'] = None
if 'consecutive_frames' not in st.session_state:
    st.session_state['consecutive_frames'] = 0
if 'frame_count' not in st.session_state:
    st.session_state['frame_count'] = 0

# Initialize pygame mixer for alarm
pygame.mixer.init()

# Define utility functions that were in utils.py
def detect_faces(frame, face_cascade):
    """Detect faces in the frame using the face cascade classifier"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces, gray

def preprocess_face_for_model(face_roi):
    """Preprocess the face ROI for model input"""
    # Resize to model input size (96x96)
    face_resized = cv2.resize(face_roi, (96, 96))
    # Convert to RGB if it's not already
    if len(face_resized.shape) == 2:  # If grayscale
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
    elif face_resized.shape[2] == 1:  # If single channel
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
    # Normalize pixel values to [0, 1]
    face_normalized = face_resized / 255.0
    # Expand dimensions to match model input shape [batch_size, height, width, channels]
    face_expanded = np.expand_dims(face_normalized, axis=0)
    return face_expanded

def check_drowsiness(prediction, consecutive_frames, threshold):
    """Check if the prediction indicates drowsiness"""
    if prediction > threshold:  # Drowsy
        consecutive_frames += 1
    else:  # Active
        consecutive_frames = max(0, consecutive_frames - 1)  # Reset or decrement
    return consecutive_frames

def draw_prediction(frame, faces, predictions, alarm_active, alarm_start_time):
    """Draw bounding boxes and drowsiness predictions on the frame"""
    for i, (x, y, w, h) in enumerate(faces):
        # Draw bounding box around face
        color = (0, 0, 255) if alarm_active else (0, 255, 0)  # Red if alarm active, green otherwise
        thickness = 3 if alarm_active else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw prediction text if available
        if i < len(predictions):
            prediction = predictions[i]
            status = "DROWSY" if prediction > 0.7 else "Active"
            confidence = prediction if prediction > 0.7 else 1 - prediction
            color = (0, 0, 255) if status == "DROWSY" else (0, 255, 0)
            
            text = f"{status}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw alarm status
    if alarm_active and alarm_start_time is not None:
        alarm_duration = time.time() - alarm_start_time
        cv2.putText(
            frame,
            f"DROWSINESS ALERT! ({alarm_duration:.1f}s)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )
    
    return frame

@st.cache_resource
def load_model(model_path="models/drowsiness_model_new.h5"):
    """Load and cache the model with custom loader to handle version incompatibilities"""
    try:
        # Try the standard loading first
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.warning(f"Standard model loading failed: {str(e)}. Trying alternative loading method...")
        
        # Specifically handle the 'batch_shape' issue with InputLayer
        if "'batch_shape'" in str(e):
            st.info("Detected 'batch_shape' compatibility issue. Using custom model loading...")
            
            # Patch the _keras_shape issue by monkey patching the layer_config function
            original_get_config = tf.keras.layers.InputLayer.get_config
            
            def patched_get_config(self):
                config = original_get_config(self)
                if 'batch_shape' in config:
                    config['batch_input_shape'] = config.pop('batch_shape')
                return config
            
            # Apply the monkey patch
            tf.keras.layers.InputLayer.get_config = patched_get_config
            
            try:
                # Try with the patched InputLayer
                model = tf.keras.models.load_model(model_path, compile=False)
                
                # Restore original method
                tf.keras.layers.InputLayer.get_config = original_get_config
                
                return model
            except Exception as patch_error:
                # Restore original method if exception occurs
                tf.keras.layers.InputLayer.get_config = original_get_config
                st.error(f"Patched loading method failed: {str(patch_error)}")
                
                # Try directly loading from H5 file using lower-level API
                try:
                    st.info("Attempting direct H5 file loading...")
                    import h5py
                    
                    # Create a simple CNN model with similar architecture
                    input_shape = (96, 96, 3)
                    model = tf.keras.Sequential([
                        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Dropout(0.25),
                        
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Dropout(0.25),
                        
                        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                        tf.keras.layers.MaxPooling2D((2, 2)),
                        tf.keras.layers.Dropout(0.25),
                        
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                    
                    # Load just the weights part
                    try:
                        # First compile the model
                        model.compile(
                            optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        # Try to load weights directly
                        model.load_weights(model_path)
                        return model
                    except Exception as weights_error:
                        st.warning(f"Direct weights loading failed: {str(weights_error)}")
                        
                        # Try with H5PY to manually extract weights
                        with h5py.File(model_path, 'r') as f:
                            # Check the structure of the H5 file
                            st.info("Examining H5 file structure...")
                            
                            # Check if it's a standard Keras model or just weights
                            if 'model_weights' in f:
                                weight_names = [name for name in f['model_weights']]
                                st.info(f"Found model weights: {weight_names}")
                                
                                # Create a very simple model as fallback
                                simple_model = tf.keras.Sequential([
                                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
                                    tf.keras.layers.MaxPooling2D((2, 2)),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')
                                ])
                                
                                simple_model.compile(
                                    optimizer='adam',
                                    loss='binary_crossentropy',
                                    metrics=['accuracy']
                                )
                                
                                # Return the simple model - it won't have the correct weights
                                # but it will allow the app to run for testing
                                st.warning("Using a simplified model without proper weights. Detection accuracy will be compromised.")
                                return simple_model
                            else:
                                st.error("Could not find model weights in the H5 file.")
                                return None
                except Exception as h5_error:
                    st.error(f"H5 loading method failed: {str(h5_error)}")
        
        # If we're here, try a complete solution that rebuilds the model
        try:
            st.info("Attempting to create a new drowsiness detection model...")
            
            # Create a standard drowsiness detection model
            new_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(96, 96, 3)),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile the model
            new_model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics=['accuracy']
            )
            
            st.warning("Created a new model since the saved model could not be loaded. This model is NOT trained and will give random predictions.")
            st.info("You need to retrain your model and save it in a TensorFlow version compatible with your local environment.")
            
            return new_model
        except Exception as final_error:
            st.error(f"Could not create fallback model: {str(final_error)}")
            return None

@st.cache_resource
def load_face_cascade():
    """Load face cascade classifier"""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def play_alarm():
    """Play alarm sound"""
    try:
        pygame.mixer.music.load("Alarm.mp3")
        pygame.mixer.music.play(loops=-1)  # Play in a loop until stopped
    except pygame.error as e:
        st.error(f"Could not play alarm sound: {str(e)}")
        # Create a fallback beeping sound using pygame
        try:
            pygame.mixer.Sound("beep.mp3").play(loops=-1)
        except:
            # If even the fallback fails, we'll use a simple beep using system sound
            import winsound
            winsound.Beep(1000, 500)  # Frequency=1000Hz, Duration=500ms

def stop_alarm():
    """Stop alarm sound"""
    try:
        pygame.mixer.music.stop()
    except pygame.error:
        pass

def process_frame(frame, model, face_cascade, threshold=0.7, frames_threshold=15):
    """Process a single frame for drowsiness detection"""
    # Ensure frame is in proper format
    if frame is None or frame.size == 0:
        return frame, 0, []
    
    # Convert frame from RGB to BGR (OpenCV format) if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame
    
    # Detect faces
    faces, gray = detect_faces(frame_bgr, face_cascade)
    
    predictions = []
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame_bgr[y:y+h, x:x+w]
        
        # Check if face ROI is valid
        if face_roi.size == 0:
            continue
        
        # Preprocess face for model input
        preprocessed_face = preprocess_face_for_model(face_roi)
        
        # Make prediction
        try:
            prediction = model.predict(preprocessed_face, verbose=0)[0][0]
            predictions.append(prediction)
            
            # Check if the person is drowsy
            st.session_state['consecutive_frames'] = check_drowsiness(
                prediction, st.session_state['consecutive_frames'], threshold
            )
            
            # Trigger alarm if drowsy for consecutive frames
            if st.session_state['consecutive_frames'] >= frames_threshold:
                if not st.session_state['alarm_status']:
                    st.session_state['alarm_status'] = True
                    st.session_state['alarm_start_time'] = time.time()
                    play_alarm()
            else:
                if st.session_state['alarm_status']:
                    st.session_state['alarm_status'] = False
                    st.session_state['alarm_start_time'] = None
                    stop_alarm()
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            predictions.append(0.5)  # Default value in case of error
    
    # Draw predictions on the frame
    result_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_with_predictions = draw_prediction(
        result_frame, 
        faces, 
        predictions, 
        st.session_state['alarm_status'], 
        st.session_state['alarm_start_time']
    )
    
    # Add drowsiness status text
    cv2.putText(
        frame_with_predictions, 
        f"Drowsiness Frames: {st.session_state['consecutive_frames']}/{frames_threshold}", 
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
    )
    
    # Increment frame count in session state
    st.session_state['frame_count'] += 1
    
    return frame_with_predictions, len(faces), predictions

def main():
    # Title and description
    st.title("Driver Drowsiness Detection System")
    st.markdown("""
    This application detects driver drowsiness in real-time using computer vision and deep learning.
    It monitors facial features to determine if a driver is getting drowsy and alerts them with an alarm.
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model path
    model_path = st.sidebar.text_input(
        "Model Path", 
        "models/drowsiness_model_new.h5",
        help="Path to the trained model file"
    )
    
    # Detection parameters
    threshold = st.sidebar.slider(
        "Drowsiness Threshold", 
        min_value=0.4, 
        max_value=0.9, 
        value=0.7, 
        step=0.05,
        help="Threshold for drowsiness prediction (higher value = more sensitive)"
    )
    
    frames_threshold = st.sidebar.slider(
        "Consecutive Frames Threshold", 
        min_value=5, 
        max_value=30, 
        value=15, 
        step=1,
        help="Number of consecutive drowsy frames to trigger alarm"
    )
    
    # Mode selection
    detection_mode = st.sidebar.radio(
        "Detection Mode",
        ["Webcam", "Upload Video", "Upload Image"]
    )
    
    # Main content
    try:
        # Display TensorFlow version info
        st.sidebar.info(f"TensorFlow version: {tf.__version__}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.info("Please make sure the model file exists in the specified path.")
            return
            
        # Load model with improved error handling
        with st.spinner("Loading model... This may take a moment."):
            model = load_model(model_path)
        
        if model is None:
            st.error("Failed to load the model. Please check the model path and format.")
            
            # Add option to continue with a dummy model for testing UI
            if st.button("Continue with untrained model for UI testing"):
                st.warning("Creating a simple untrained model for UI testing only. Predictions will be random.")
                # Create a simple model for UI testing
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(96, 96, 3)),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
            else:
                return
            
        # Load face cascade
        face_cascade = load_face_cascade()
        
        # Check if alarm sound file exists
        if not os.path.exists("Alarm.mp3"):
            st.sidebar.warning("Alarm sound file not found. A default beep will be used instead.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Different detection modes
            if detection_mode == "Webcam":
                st.subheader("Webcam Feed")
                run_button = st.button("Start/Stop Camera")
                
                if "camera_running" not in st.session_state:
                    st.session_state["camera_running"] = False
                
                if run_button:
                    st.session_state["camera_running"] = not st.session_state["camera_running"]
                
                if st.session_state["camera_running"]:
                    # Create a placeholder for the webcam feed
                    video_placeholder = st.empty()
                    stop_button_placeholder = st.empty()
                    
                    if stop_button_placeholder.button("Stop Camera"):
                        st.session_state["camera_running"] = False
                        stop_alarm()
                        st.experimental_rerun()
                    
                    # Start webcam
                    cap = cv2.VideoCapture(0)
                    
                    if not cap.isOpened():
                        st.error("Could not open webcam! Please check your camera connection.")
                    else:
                        while st.session_state["camera_running"]:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Could not read from webcam!")
                                break
                            
                            # Convert BGR to RGB for display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Process frame
                            processed_frame, face_count, predictions = process_frame(
                                frame_rgb, model, face_cascade, threshold, frames_threshold
                            )
                            
                            # Display processed frame
                            video_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                            
                            # Add a small delay
                            time.sleep(0.01)
                        
                        # Release resources
                        cap.release()
                        stop_alarm()
                else:
                    st.info("Click 'Start Camera' to begin real-time drowsiness detection.")
            
            elif detection_mode == "Upload Video":
                st.subheader("Video Upload")
                uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
                
                if uploaded_video is not None:
                    # Save uploaded video to a temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    temp_file.write(uploaded_video.read())
                    video_path = temp_file.name
                    temp_file.close()
                    
                    # Create a placeholder for the video feed
                    video_placeholder = st.empty()
                    
                    # Process button
                    if st.button("Process Video"):
                        # Reset session state
                        st.session_state['consecutive_frames'] = 0
                        st.session_state['alarm_status'] = False
                        
                        # Start video capture
                        cap = cv2.VideoCapture(video_path)
                        
                        if not cap.isOpened():
                            st.error("Could not open the video file!")
                        else:
                            # Get video properties
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_delay = 1 / fps if fps > 0 else 0.03  # Default to 30fps if fps is 0
                            
                            # Process each frame
                            progress_bar = st.progress(0)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            
                            frame_count = 0
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                # Convert BGR to RGB for display
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Process frame
                                processed_frame, face_count, predictions = process_frame(
                                    frame_rgb, model, face_cascade, threshold, frames_threshold
                                )
                                
                                # Display processed frame
                                video_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                                
                                # Update progress bar
                                frame_count += 1
                                progress_value = min(frame_count / total_frames, 1.0) if total_frames > 0 else 0
                                progress_bar.progress(progress_value)
                                
                                # Add delay to match video speed
                                time.sleep(frame_delay)
                            
                            # Release resources
                            cap.release()
                            stop_alarm()
                            
                            # Clean up temporary file
                            os.unlink(video_path)
                            st.success("Video processing completed!")
                    
                    st.info("Note: Uploaded videos will be processed and displayed here.")
            
            elif detection_mode == "Upload Image":
                st.subheader("Image Upload")
                uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
                
                if uploaded_image is not None:
                    # Read image
                    image = Image.open(uploaded_image)
                    image_np = np.array(image)
                    
                    # Process image
                    processed_image, face_count, predictions = process_frame(
                        image_np, model, face_cascade, threshold, frames_threshold
                    )
                    
                    # Display processed image
                    st.image(processed_image, caption="Processed Image", use_column_width=True)
                    
                    # Display results
                    if face_count > 0:
                        st.success(f"Detected {face_count} {'face' if face_count == 1 else 'faces'}")
                        
                        for i, pred in enumerate(predictions):
                            status = "Active" if pred < threshold else "Drowsy"
                            confidence = 1 - pred if pred < threshold else pred
                            st.info(f"Face {i+1}: {status} (Confidence: {confidence:.2f})")
                    else:
                        st.warning("No faces detected in the image.")
        
        with col2:
            # Display model information
            st.subheader("Model Information")
            model_loaded = model is not None
            st.success(f"Model Status: {'Loaded' if model_loaded else 'Not Loaded'}")
            
            if model_loaded:
                st.write("Model Architecture:")
                try:
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.code("\n".join(model_summary))
                except Exception as e:
                    st.warning(f"Could not display model summary: {str(e)}")
                    st.write("Model layers:")
                    for i, layer in enumerate(model.layers):
                        st.write(f"Layer {i}: {layer.__class__.__name__} - Output Shape: {layer.output_shape}")
            
            # Display system statistics
            st.subheader("System Statistics")
            stats_container = st.container()
            
            # Update stats periodically
            stats_placeholder = stats_container.empty()
            stats_data = {
                "Detection Threshold": threshold,
                "Consecutive Frames Required": frames_threshold,
                "Current Consecutive Frames": st.session_state['consecutive_frames'],
                "Alarm Status": "ACTIVE" if st.session_state['alarm_status'] else "Inactive",
                "Processed Frames": st.session_state['frame_count']
            }
            
            # Display stats as a table
            stats_df = pd.DataFrame(list(stats_data.items()), columns=["Metric", "Value"])
            stats_placeholder.table(stats_df)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        if st.session_state.get('alarm_status', False):
            stop_alarm()

if __name__ == "__main__":
    main()
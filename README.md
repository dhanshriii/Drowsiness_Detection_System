# Driver Drowsiness Detection System

This project implements a real-time drowsiness detection system using computer vision and deep learning. The system monitors a driver's face through a webcam and alerts them when signs of drowsiness are detected.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd driver-drowsiness-detection
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the Alarm.mp3 file in the project directory

## Usage

### Training the model
```bash
python model.py
```

### Real-time detection using webcam
```bash
python realtime_detection.py --model models/drowsiness_detection_final.h5
```

### Running the Streamlit web application
```bash
streamlit run app.py
```

## Features

- Real-time drowsiness detection through webcam feed
- Support for uploaded videos and images
- Audible and visual alerts when drowsiness is detected
- Adjustable sensitivity parameters
- Web interface using Streamlit
- Detailed statistics and model information display

## Project Structure

```
drowsiness-detection/
│
├── app.py                  # Streamlit web application
├── model.py                # Model definition and training
├── realtime_detection.py   # Stand-alone real-time detection script
├── utils.py                # Helper functions
├── models/                 # Directory for trained models
├── Alarm.mp3               # Alarm sound file
└── requirements.txt        # List of dependencies
```

## Dataset Structure

The model is trained on a dataset with the following structure:
```
A:/TY/Sem II/AIML/Driver-Drowsiness-Detection/Data/0 FaceImages/
│
├── Active Subjects/        # Images of alert drivers
└── Fatigue Subjects/       # Images of drowsy drivers
```

## How It Works

1. **Face Detection**: OpenCV's Haar Cascade classifier is used to detect faces in each frame.
2. **Feature Extraction**: The detected face is preprocessed and fed into the CNN model.
3. **Drowsiness Prediction**: The model predicts whether the driver is drowsy or alert.
4. **Alert System**: If drowsiness is detected for several consecutive frames, an alert is triggered.

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:
- 4 convolutional layers with max-pooling
- Flattening layer
- Dense layer with dropout for regularization
- Output layer with sigmoid activation for binary classification

## Customization

You can adjust various parameters through the Streamlit UI or command-line arguments:
- Drowsiness threshold (sensitivity)
- Number of consecutive frames to trigger alert
- Model path
- Video source (webcam, uploaded video, or image)
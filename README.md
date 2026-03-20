# Multimodal Deepfake Detection System

## Overview

This project detects deepfake videos using both visual and audio information. It combines features extracted from video frames and audio signals to improve accuracy compared to single-modality models.

## Features

* Video-based detection using CNN (ResNet18)
* Audio-based detection using spectrograms
* Multimodal feature fusion
* Explainability using Grad-CAM
* Simple web interface for uploading and testing videos

## Tech Stack

* Python
* PyTorch
* OpenCV
* Flask
* NumPy, Pandas
* Matplotlib
* Librosa

## Project Structure

* final_app/ → main application logic
* server/ → backend (Flask server)
* model_resnet18/ → model architecture
* templates/ → HTML files
* static/ → CSS, JS, UI assets
* saved_models/ → trained model weights

## Setup Instructions

1. Clone the repository:
   git clone <your-repo-link>
   cd <repo-name>

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   python server.py

4. Open in browser:
   http://127.0.0.1:5000/

## Model Weights

If model files are not included in the repository, download them from: <your-drive-link>

Place the files inside the `saved_models/` directory.

## Results

* Achieved approximately 92% accuracy
* Outperformed single-modality models
* Provides visual explanations using Grad-CAM

## Future Improvements

* Real-time deepfake detection
* Improved audio-video synchronization
* Training on larger and more diverse datasets

## Author

Your Name

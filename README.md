# Action Recognition in Sports Videos

This project provides a web-based system for recognizing actions in sports videos, such as "shooting," "passing," and "dribbling" in basketball. It uses a **FastAPI** backend to process videos with a pre-trained I3D model and an LSTM model, and a **Flask** frontend to allow users to upload videos and view results through a web interface.

## Features

- Upload sports videos (MP4 or AVI) via a web interface.
- Detect actions using I3D for feature extraction and LSTM for sequence modeling.
- Display recognized actions and the uploaded video in the browser.
- Memory management to prevent GPU and disk space leaks.
- Logging for debugging and monitoring.

## Project Structure

action_recognition/├── app.py # Flask frontend for web interface├── api.py # FastAPI backend for video processing├── templates/│ └── index.html # HTML template for frontend├── static/│ ├── css/│ │ └── style.css # CSS for styling│ └── uploads/ # Temporary storage for uploaded videos├── temp_uploads/ # Temporary storage for FastAPI video processing├── requirements.txt # Python dependencies└── README.md # This file

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (e.g., RTX 4080 SUPER) with CUDA and cuDNN for TensorFlow
- FFmpeg installed for video processing (`sudo apt-get install ffmpeg` on Ubuntu)
- A trained LSTM model weights file (`lstm_model_weights.h5`) or a dataset to train the model (e.g., UCF101)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd action_recognition
   ```

Create a Virtual Environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install Dependencies:
pip install -r requirements.txt

Contents of requirements.txt:
tensorflow
tensorflow-hub
opencv-python
ffmpeg-python
fastapi
uvicorn
numpy
flask
requests

Install FFmpeg (if not already installed):

On Ubuntu:sudo apt-get install ffmpeg

On macOS:brew install ffmpeg

On Windows: Download from FFmpeg website and add to PATH.

Prepare the LSTM Model:

If you have a trained model, place lstm_model_weights.h5 in the project root and uncomment the following line in api.py:lstm_model.load_weights("lstm_model_weights.h5")

If not, train the model using a dataset like UCF101. See Training the Model.

Running the Application

Start the FastAPI Backend:
uvicorn api:app --reload --port 8000

The backend will run on http://127.0.0.1:8000.

Start the Flask Frontend:
python app.py

The frontend will run on http://127.0.0.1:5000.

Access the Web Interface:

Open http://127.0.0.1:5000 in your browser.
Upload a sports video (MP4 recommended, AVI supported) to analyze actions.
View the recognized actions (e.g., "shooting," "passing," "dribbling") and the uploaded video.

Usage

Uploading a Video:

Use the web interface to select a video file (e.g., test_video.mp4).
Click "Upload and Analyze" to process the video.
The results will show the top recognized actions and the video in a player.

Sample Video:

Download a video from the UCF101 dataset (http://crcv.ucf.edu/data/UCF101.php), e.g., v_Basketball_g01_c01.avi.
Convert to MP4 for browser compatibility:ffmpeg -i v_Basketball_g01_c01.avi -c:v libx264 -c:a aac -strict -2 test_video.mp4

Testing with Postman (Optional):

Send a POST request to http://127.0.0.1:8000/recognize_action with a form-data body:
Key: video, Value: Select your video file.

Check the JSON response for actions.

Training the Model
If you don’t have a pre-trained lstm_model_weights.h5, train the LSTM model:

Download a dataset like UCF101.
Extract I3D features for video clips using the get_i3d_features function.
Prepare data with shape (num_samples, num_clips, feature_dim) for features and (num_samples, num_classes) for labels.
Train the model:from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
lstm_model.save_weights("lstm_model_weights.h5")

Update ACTION_LABELS in api.py to match your dataset’s classes.

Troubleshooting

Video Not Displaying:

Cause: The video file may be deleted before rendering or in an unsupported format.
Fix:
Ensure the video is MP4 with H.264 codec:ffmpeg -i input.avi -c:v libx264 -c:a aac -strict -2 output.mp4

Check Flask logs for file creation/deletion:INFO:**main**:Saved video to: static/uploads/...

Inspect the browser’s Network tab for the video URL (e.g., http://127.0.0.1:5000/static/uploads/1634567890_test_video.mp4).
Increase cleanup time in app.py:if file_age > 600: # 10 minutes

"List index out of range" Error:

Cause: Mismatch between ACTION_LABELS and NUM_CLASSES.
Fix: Ensure ACTION_LABELS in api.py matches the model’s output classes:ACTION_LABELS = ["shooting", "passing", "dribbling"]

"No valid clips extracted" Error:

Cause: Video is too short (<64 frames, ~2 seconds at 30 fps).
Fix:
Use a longer video or reduce clip_length in api.py:clips = extract_clips(video_path, clip_length=32, step=16)

Verify video readability:cap = cv2.VideoCapture("test_video.mp4")
print(cap.isOpened()) # Should print True
cap.release()

Memory Issues:

Monitor GPU memory with nvidia-smi.
Check disk space in static/uploads and temp_uploads.
The code includes memory cleanup (tf.keras.backend.clear_session(), gc.collect(), file deletion).

Notes

Model: The LSTM model must be trained or loaded with weights for accurate predictions. Update ACTION_LABELS to match your dataset.
Video Format: MP4 with H.264 is recommended for browser compatibility. Convert AVI files using FFmpeg.
Performance: For large videos, reduce clip_length or optimize preprocessing.
Production: Use Gunicorn for Flask (gunicorn -w 4 app:app -b 0.0.0.0:5000) and Uvicorn workers for FastAPI. Consider a reverse proxy (e.g., Nginx).

License
MIT License. See LICENSE for details.
Acknowledgments

UCF101 Dataset for sample videos.
TensorFlow Hub for the I3D model.
FastAPI and Flask for backend and frontend frameworks.

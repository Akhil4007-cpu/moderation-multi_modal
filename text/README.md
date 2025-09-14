Video Text Moderation System

A Python-based system to detect unsafe content in videos using OCR (pytesseract) and speech recognition (SpeechRecognition + ffmpeg). It extracts text from video frames and audio, checks for unsafe keywords, and classifies content as Safe or Unsafe.

Features

Text Extraction from Video Frames using Tesseract OCR.

Audio Transcription from video using ffmpeg + Google SpeechRecognition.

Unsafe Word Detection with a predefined list of keywords (can be expanded).

Text Classification using a pre-trained Naive Bayes / ML model (text_model.pkl).

Outputs full extracted text, unsafe words, category, and confidence score.

Folder Structure

project/
│
├─ Dataset/
│ ├─ Safe_Text_Dataset/
│ │ ├─ train/
│ │ └─ test/
│ ├─ Violence_Terrorism_Dataset/
│ │ ├─ train/
│ │ └─ test/
│ └─ ... other categories ...
│
├─ train_text_classifier.py # Script to train the text classification model
├─ test_text_classifier.py # Script to test the model on test data
├─ video_text_moderation.py # Run on a video file to detect unsafe content
├─ text_model.pkl # Saved trained model (after training)
└─ README.md

Prerequisites

Python 3.12

NumPy 2.2.6 (or compatible)

OpenCV, pytesseract, joblib, scikit-learn, SpeechRecognition, pandas

FFmpeg installed and added to system PATH

Install dependencies:

pip install opencv-python pytesseract joblib scikit-learn SpeechRecognition pandas

Install Tesseract OCR from https://github.com/tesseract-ocr/tesseract
 and update the path in scripts if necessary.

Usage
1. Train the Model

python train_text_classifier.py

Reads all text files from Dataset/<Category>/train/.

Converts text into features (TF-IDF or similar) and trains a Naive Bayes classifier.

Saves the trained model as text_model.pkl.

2. Test the Model

python test_text_classifier.py

Reads text files from Dataset/<Category>/test/.

Loads text_model.pkl and makes predictions.

Saves test results in results.json.

Prints accuracy and classification report.

3. Run Video Moderation

python video_text_moderation.py

Input the path of the video.

The script will:

Extract text from video frames (OCR)

Extract and transcribe audio (SpeechRecognition)

Merge texts and detect unsafe words

Output full extracted text, unsafe words, category, and confidence

Example output:

📜 Full Extracted Text:
The formulas problems example...

🚨 Unsafe Words Found: ['kill', 'drugs']

Safe or Unsafe: Unsafe

Confidence: 0.99

Customize Unsafe Words

Edit UNSAFE_KEYWORDS in video_text_moderation.py:

UNSAFE_KEYWORDS = ["suicide", "murder", "kill", "drugs", "weapon", "crime", ...]

Notes

If audio has no speech, the system uses video frame text as backup.

Confidence is:

1.0 if no unsafe words found (Safe)

0.99 if unsafe words found (Unsafe)

Designed for Python 3.12 + NumPy 2.2.6 without Whisper / PyTorch to avoid version conflicts.

License

This project is open-source and free to use for educational purposes.
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ----------------- CONFIG -----------------
DATASET_PATH = "C:/Users/Harshitha Addanki/Desktop/audio/dataset"
CATEGORIES = ["adult", "drugs", "hate", "safe", "spam", "violence"]

# ----------------- FEATURE EXTRACTION -----------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)  # load audio
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    # 2. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    # 3. Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)
    # Combine features
    features = np.concatenate([mfcc_mean, chroma_mean, contrast_mean])
    return features

# ----------------- LOAD DATA -----------------
print("Loading training data...")
X = []
y = []

for idx, cat in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_PATH, "train", cat)
    print(f"Processing category: {cat}")
    files = [f for f in os.listdir(folder) if f.lower().endswith(".mp3")]
    print(f"Found {len(files)} files in {cat}")
    
    for i, file in enumerate(files):
        if i % 10 == 0:  # Progress indicator
            print(f"  Processing file {i+1}/{len(files)}")
        path = os.path.join(folder, file)
        features = extract_features(path)
        X.append(features)
        y.append(idx)

X = np.array(X)
y = np.array(y)
print(f"Training data shape: {X.shape}")

# ----------------- TRAIN / TEST SPLIT -----------------
print("Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ----------------- TRAIN CLASSIFIER -----------------
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
print("Training completed!")

# ----------------- EVALUATE -----------------
print("Evaluating model...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[✅] Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# ----------------- SAVE MODEL -----------------
print("Saving model...")
joblib.dump(clf, "audio_classifier_sklearn.pkl")
print("[✅] Model saved as audio_classifier_sklearn.pkl")
print("Training completed successfully!")

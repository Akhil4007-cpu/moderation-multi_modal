# 🎵 Audio Moderation System

A machine learning-based audio content moderation system that classifies audio and video files into safe/unsafe categories with high accuracy.

## 📊 Performance Metrics

- **Overall Accuracy: 94.17%**
- **Safe Content Detection: 95.0%**
- **Unsafe Content Detection: 100.0%**
- **Categories: adult, drugs, hate, safe, spam, violence**

## 🚀 Features

- ✅ **Audio & Video Support** - Handles MP3, MP4, AVI, MOV files
- ✅ **Real-time Classification** - Instant safe/unsafe determination
- ✅ **Confidence Scores** - Detailed probability breakdowns
- ✅ **JSON Output** - Structured results for integration
- ✅ **Batch Processing** - Test multiple files at once
- ✅ **High Accuracy** - 94%+ classification accuracy

## 📁 Project Structure

```
audio/
├── train_audio_classifier.py          # 🎓 TRAINING - Creates the model
├── batch_moderation.py                # 📊 TESTING - Tests the model
├── interactive_moderation.py          # 🛡️ MODERATION - Uses model on new files
├── run_training.bat                   # Training batch script
├── audio_classifier_sklearn.pkl       # Trained model (2.9MB)
├── moderation_results.json            # Individual moderation results
├── batch_moderation_results.json      # Batch test results
└── dataset/
    ├── train/                         # Training data (479 files)
    └── test/                          # Testing data (120 files)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Required packages (install via pip):

```bash
pip install librosa numpy scikit-learn joblib pydub
```

### Quick Setup
1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. The trained model is already included (`audio_classifier_sklearn.pkl`)

## 🎯 Usage

### 🛡️ MODERATION (Main Use)

Run the interactive system that asks for file paths:

```bash
python interactive_moderation.py
```

Then enter your file path when prompted:
```
🎵 File path: dataset/test/safe/point_10.mp3
```

### 📊 TESTING

Test all files in the dataset:

```bash
python batch_moderation.py
```

### 🎓 TRAINING

To train the model with your data:

```bash
python train_audio_classifier.py
```

**Example Output:**
```
🎵 AUDIO MODERATION RESULTS
============================================================
📁 File: example.mp3
🏷️  Category: safe
📊 Confidence: 67.5%
🛡️  Safety Status: SAFE
⏰ Timestamp: 2025-09-14T14:46:05.118872

📈 All Category Probabilities:
     adult: 3.5%
     drugs: 3.5%
     hate: 10.0%
  ✅ safe: 67.5%
     spam: 3.5%
     violence: 12.0%
============================================================
```

### Batch Testing

Test all files in the dataset:

```bash
python batch_moderation.py
```

**Example Output:**
```
📊 BATCH MODERATION SUMMARY
======================================================================
📁 Total files processed: 120
🎯 Overall accuracy: 94.17%

📈 Category-wise Results:
     adult: 18/20 ( 90.0%)
     drugs: 20/20 (100.0%)
      hate: 20/20 (100.0%)
      safe: 19/20 ( 95.0%)
      spam: 18/20 ( 90.0%)
  violence: 18/20 ( 90.0%)

🛡️  Safety Analysis:
  ✅ Safe files correctly identified: 19/20 (95.0%)
  ❌ Unsafe files correctly identified: 100/100 (100.0%)
======================================================================
```


## 📄 JSON Output Format

### Individual Moderation Results (`moderation_results.json`)

```json
{
  "moderation_history": [
    {
      "file_path": "dataset/test/safe/point_10.mp3",
      "file_name": "point_10.mp3",
      "predicted_category": "safe",
      "confidence": 67.5,
      "is_safe": true,
      "safety_status": "SAFE",
      "all_probabilities": {
        "adult": 3.5,
        "drugs": 3.5,
        "hate": 10.0,
        "safe": 67.5,
        "spam": 3.5,
        "violence": 12.0
      },
      "timestamp": "2025-09-14T14:46:05.118872",
      "model_used": "audio_classifier_sklearn.pkl"
    }
  ],
  "last_updated": "2025-09-14T14:46:18.541691",
  "total_moderations": 1
}
```

### Batch Results (`batch_moderation_results.json`)

```json
{
  "moderation_summary": {
    "total_files_processed": 120,
    "overall_accuracy": 94.17,
    "safe_files_correctly_identified": 19,
    "safe_files_total": 20,
    "safe_accuracy": 95.0,
    "unsafe_files_correctly_identified": 100,
    "unsafe_files_total": 100,
    "unsafe_accuracy": 100.0,
    "timestamp": "2025-09-14T14:45:53.835857",
    "model_used": "audio_classifier_sklearn.pkl"
  },
  "category_statistics": {
    "adult": {
      "total_files": 20,
      "correct_predictions": 18,
      "safe_predictions": 0,
      "unsafe_predictions": 20
    }
  },
  "detailed_results": [...]
}
```

## 🎵 Supported File Formats

### Audio Files
- MP3
- WAV
- FLAC
- M4A

### Video Files
- MP4
- AVI
- MOV
- MKV

*Note: Video files are automatically converted to audio for processing*

## 🏷️ Content Categories

| Category | Description | Safety Status |
|----------|-------------|---------------|
| **safe** | Appropriate content | ✅ SAFE |
| **adult** | Adult content | ❌ UNSAFE |
| **drugs** | Drug-related content | ❌ UNSAFE |
| **hate** | Hate speech | ❌ UNSAFE |
| **spam** | Spam content | ❌ UNSAFE |
| **violence** | Violent content | ❌ UNSAFE |

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Features**: MFCC, Chroma, Spectral Contrast
- **Training Data**: 479 audio files
- **Test Data**: 120 audio files
- **Feature Dimensions**: 59 features per audio file

### Feature Extraction
1. **MFCC (Mel-frequency cepstral coefficients)**: 40 features
2. **Chroma**: 12 features  
3. **Spectral Contrast**: 7 features

### Performance by Category
- **Drugs**: 100% accuracy
- **Hate**: 100% accuracy
- **Safe**: 95% accuracy
- **Adult**: 90% accuracy
- **Spam**: 90% accuracy
- **Violence**: 90% accuracy

## 🚨 Safety Analysis

The system excels at identifying unsafe content:
- **100% accuracy** in detecting unsafe content
- **95% accuracy** in detecting safe content
- **Zero false negatives** for unsafe content
- **Low false positives** for safe content

## 📈 Integration Examples

### Python Integration
```python
import joblib
import librosa
import numpy as np

# Load model
model = joblib.load('audio_classifier_sklearn.pkl')

# Extract features (same as in the system)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features = np.concatenate([np.mean(mfcc.T, axis=0),
                               np.mean(chroma.T, axis=0),
                               np.mean(contrast.T, axis=0)])
    return features

# Make prediction
features = extract_features('your_audio.mp3')
prediction = model.predict(features.reshape(1, -1))[0]
confidence = model.predict_proba(features.reshape(1, -1))[0].max() * 100
```

## 🛡️ Use Cases

- **Content Moderation**: Automatically filter inappropriate audio/video content
- **Social Media**: Moderate user-uploaded content
- **Streaming Platforms**: Real-time content classification
- **Educational Platforms**: Ensure safe learning environments
- **Corporate Communications**: Filter meeting recordings

## ⚠️ Limitations

- Model trained on specific dataset - may need retraining for different domains
- Audio quality affects accuracy
- Very short audio clips (< 1 second) may have reduced accuracy
- Background noise can impact classification

## 🔄 Retraining

To retrain the model with new data:

1. Update the `DATASET_PATH` in `train_audio_classifier.py`
2. Ensure your dataset follows the same structure:
   ```
   dataset/
   ├── train/
   │   ├── adult/
   │   ├── drugs/
   │   ├── hate/
   │   ├── safe/
   │   ├── spam/
   │   └── violence/
   └── test/
       ├── adult/
       ├── drugs/
       ├── hate/
       ├── safe/
       ├── spam/
       └── violence/
   ```
3. Run: `python train_audio_classifier.py`

## 📞 Support

For issues or questions:
1. Check the JSON output for detailed error information
2. Ensure all dependencies are installed
3. Verify file paths and permissions
4. Check audio file format compatibility

## 📄 License

This project is for educational and research purposes. Please ensure compliance with local laws and regulations when using for content moderation.

---

**🎉 Ready to moderate audio content with 94%+ accuracy!**

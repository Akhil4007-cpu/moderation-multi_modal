# Multi-Modal Content Moderation Integration Guide

## Overview

This guide explains the unified multi-modal content moderation system that integrates video, image, audio, and text processing capabilities into a single comprehensive solution.

## Architecture

### System Components

1. **Video/Image Processing**: YOLO-based detection models for visual content
2. **Audio Processing**: Machine learning classifiers for audio content moderation
3. **Text Processing**: OCR, speech recognition, and keyword-based text analysis
4. **Unified Processor**: Central orchestrator that combines all modalities

### Integration Flow

```
Input File → File Type Detection → Multi-Modal Processing → Unified Assessment
    ↓              ↓                      ↓                    ↓
  Any File    Auto-detect or        Process with all       Combined
  Format      Manual mode          relevant modalities    Safety Score
```

## Unified Multi-Modal Processor

### Features

- **Auto-detection**: Automatically determines file type and processing mode
- **Comprehensive Analysis**: Processes video files with frame analysis, audio extraction, and text extraction
- **Unified Assessment**: Combines results from all modalities for overall safety assessment
- **Flexible Modes**: Support for specific modality processing

### Usage Examples

#### Basic Usage
```bash
# Auto-detect and process any file
python unified_multimodal_processor.py video.mp4
python unified_multimodal_processor.py image.jpg
python unified_multimodal_processor.py audio.mp3
```

#### Advanced Usage
```bash
# Comprehensive video analysis (all modalities)
python unified_multimodal_processor.py video.mp4 --mode video_comprehensive

# Specific modality processing
python unified_multimodal_processor.py video.mp4 --mode audio
python unified_multimodal_processor.py video.mp4 --mode text

# Custom confidence threshold
python unified_multimodal_processor.py video.mp4 --conf 0.7

# Save to specific output file
python unified_multimodal_processor.py video.mp4 --output analysis_results.json
```

## Processing Modes

### 1. Image Mode
- Processes single images with all YOLO models
- Detects: accidents, fights, weapons, fire, NSFW content, objects
- Output: Detection results with bounding boxes and confidence scores

### 2. Video Comprehensive Mode
- **Frame Analysis**: Extracts frames and runs YOLO detection
- **Audio Analysis**: Extracts audio and runs audio classifier
- **Text Analysis**: OCR on frames + speech-to-text on audio
- **Unified Assessment**: Combines all results for overall safety score

### 3. Audio Mode
- Extracts audio features (MFCC, chroma, spectral contrast)
- Classifies into categories: adult, drugs, hate, safe, spam, violence
- Supports both audio files and video files (extracts audio)

### 4. Text Mode
- Processes text files or extracted text content
- Keyword-based unsafe content detection
- ML classification (when model available)

## Output Format

### Unified Assessment Structure
```json
{
  "modality": "video_comprehensive",
  "file_path": "path/to/video.mp4",
  "timestamp": "2024-01-15T10:30:00",
  "video_analysis": {
    "frames_analyzed": 5,
    "frame_results": [...]
  },
  "audio_analysis": {
    "predicted_category": "safe",
    "confidence": 95.2,
    "safety_status": "SAFE"
  },
  "text_analysis": {
    "unsafe_words_found": [],
    "safety_status": "SAFE"
  },
  "unified_assessment": {
    "overall_safety": "SAFE",
    "confidence_score": 92.5,
    "unsafe_modalities": [],
    "summary": "Content appears safe across all modalities."
  }
}
```

## Model Requirements

### YOLO Models (Video/Image)
- Accident detection: `models/accident/weights/yolov8s.pt`
- Fight detection: `models/fight/weights/nano_weights.pt`, `small_weights.pt`
- Weapon detection: `models/weapon/weights/weapon_detection.pt`
- Fire detection: `models/fire/weights/yolov8n.pt`, `yolov8s.pt`
- NSFW detection: `models/nsfw/weights/classification_model.pt`, `segmentation_model.pt`
- Object detection: `yolov8n.pt`

### Audio Model
- Audio classifier: `audio/audio_classifier_sklearn.pkl`
- Categories: adult, drugs, hate, safe, spam, violence

### Text Model
- Text classifier: `text/text_model.pkl` (optional)
- Keyword-based detection with predefined unsafe keywords

## Dependencies

### Core Requirements
```bash
# Video/Image processing
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0

# Audio processing
librosa>=0.10.0
pydub>=0.25.0
soundfile>=0.12.0

# Text processing
pytesseract>=0.3.10
SpeechRecognition>=3.10.0

# Machine learning
scikit-learn>=1.3.0
joblib>=1.3.0
```

### External Dependencies
- **FFmpeg**: Required for video/audio processing
- **Tesseract OCR**: Required for text extraction from images
- **PyTorch**: Required for YOLO models

## Installation

### Quick Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch (choose appropriate version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install external dependencies
# Windows: Download FFmpeg and Tesseract, add to PATH
# Linux: sudo apt install ffmpeg tesseract-ocr
```

### Verification
```bash
# Test all components
python -c "import ultralytics, cv2, torch, librosa, pytesseract; print('✅ All packages imported successfully')"
ffmpeg -version
tesseract --version
```

## Performance Considerations

### Processing Speed
- **Image Mode**: ~1-3 seconds per image (9 models)
- **Video Comprehensive**: ~5-15 seconds for 5-second video
- **Audio Mode**: ~2-5 seconds per minute of audio
- **Text Mode**: ~1-2 seconds per document

### Memory Usage
- YOLO models: ~500MB-1GB RAM per model
- Audio processing: ~200-500MB RAM
- Text processing: ~100-200MB RAM

### Optimization Tips
1. Use GPU acceleration for YOLO models when available
2. Process shorter video segments for faster results
3. Adjust confidence thresholds to reduce false positives
4. Use specific modes instead of comprehensive when only one modality is needed

## Integration Examples

### Batch Processing
```python
import os
from pathlib import Path
from unified_multimodal_processor import UnifiedMultiModalProcessor

processor = UnifiedMultiModalProcessor()

# Process all files in a directory
for file_path in Path("media_files").glob("*"):
    if file_path.suffix.lower() in ['.mp4', '.jpg', '.png', '.mp3', '.wav']:
        results = processor.process_file(file_path)
        print(f"Processed {file_path.name}: {results['unified_assessment']['overall_safety']}")
```

### API Integration
```python
from flask import Flask, request, jsonify
from unified_multimodal_processor import UnifiedMultiModalProcessor

app = Flask(__name__)
processor = UnifiedMultiModalProcessor()

@app.route('/moderate', methods=['POST'])
def moderate_content():
    file_path = request.json['file_path']
    results = processor.process_file(file_path)
    return jsonify(results)
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure all model files are present in correct directories
   - Check file permissions and paths

2. **FFmpeg Not Found**
   - Install FFmpeg and add to system PATH
   - Verify with `ffmpeg -version`

3. **Tesseract Not Found**
   - Install Tesseract OCR
   - Update path in code if needed: `pytesseract.pytesseract.tesseract_cmd`

4. **Audio Processing Errors**
   - Install audio codecs: `pip install soundfile`
   - Check audio file format compatibility

5. **Memory Issues**
   - Reduce batch size or process files individually
   - Use CPU-only mode if GPU memory is limited

### Debug Mode
```bash
# Run with verbose output
python unified_multimodal_processor.py video.mp4 --verbose

# Check individual modalities
python unified_multimodal_processor.py video.mp4 --mode video
python unified_multimodal_processor.py video.mp4 --mode audio
python unified_multimodal_processor.py video.mp4 --mode text
```

## Future Enhancements

### Planned Features
1. **Real-time Processing**: Stream processing capabilities
2. **Advanced Text Analysis**: Sentiment analysis and context understanding
3. **Enhanced Audio Models**: Emotion detection and speaker identification
4. **Temporal Analysis**: Cross-frame consistency for video content
5. **Custom Model Training**: Tools for training domain-specific models

### Performance Improvements
1. **Model Optimization**: Quantization and pruning for faster inference
2. **Parallel Processing**: Multi-threading for concurrent modality processing
3. **Caching**: Result caching for repeated content analysis
4. **Streaming**: Chunk-based processing for large files

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review the main README.md for general setup
3. Check individual modality documentation in respective folders
4. Create issues in the project repository

---

This integration guide provides comprehensive information for using the unified multi-modal content moderation system. The system is designed to be flexible, extensible, and production-ready for various content moderation scenarios.

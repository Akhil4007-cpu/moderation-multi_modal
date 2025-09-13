# Video Frame Processor Guide

## Overview
The Video Frame Processor extracts frames from videos at 1-second intervals and tests each frame with all available AI models for comprehensive content moderation. It automatically cleans up extracted frames after processing.

## Features
- ✅ **FFmpeg Integration**: Extracts 1 frame per second from any video format
- ✅ **Multi-Model Testing**: Tests each frame with all available models (accident, fight, fire, nsfw, weapon, objects)
- ✅ **Clear Results Display**: Detailed console output showing detections per frame and model
- ✅ **Automatic Cleanup**: Removes extracted frames after processing (unless specified otherwise)
- ✅ **Comprehensive Reports**: Generates detailed JSON results and summary reports
- ✅ **Safety Assessment**: Overall safety rating based on detected content

## Prerequisites

### 1. Install FFmpeg
**Windows:**
- Download from https://ffmpeg.org/download.html
- Extract to a folder (e.g., `C:\ffmpeg`)
- Add `C:\ffmpeg\bin` to your system PATH
- Verify installation: `ffmpeg -version`

**Alternative (using Chocolatey):**
```bash
choco install ffmpeg
```

### 2. Python Dependencies
```bash
pip install -r integrated_yolo_runner/requirements.txt
```

## Usage

### Basic Usage
```bash
python video_frame_processor.py path/to/your/video.mp4
```

### Advanced Options
```bash
# Custom output directory
python video_frame_processor.py video.mp4 --output ./my_results

# Custom confidence threshold
python video_frame_processor.py video.mp4 --conf 0.5

# Keep extracted frames (don't auto-cleanup)
python video_frame_processor.py video.mp4 --keep-frames

# Custom temporary directory for frames
python video_frame_processor.py video.mp4 --temp ./temp_frames
```

### Full Command Options
```bash
python video_frame_processor.py VIDEO_PATH [OPTIONS]

Arguments:
  VIDEO_PATH                Path to the video file

Options:
  --output, -o DIR         Output directory for results
  --temp, -t DIR           Temporary directory for extracted frames
  --conf, -c FLOAT         Confidence threshold (default: 0.25)
  --keep-frames            Keep extracted frames after processing
  --help                   Show help message
```

## Example Workflow

1. **Place your video** in any accessible location
2. **Run the processor**:
   ```bash
   python video_frame_processor.py "C:\Videos\test_video.mp4"
   ```
3. **Watch the real-time output**:
   ```
   🎥 Video: C:\Videos\test_video.mp4
   📁 Output: d:\akhil\video_analysis_results
   🗂️  Temp frames: C:\Users\...\AppData\Local\Temp\video_frames_xyz

   📊 Video Info: 30.00 FPS, 120.50s, 3615 frames
   🔄 Extracting frames (1 per second)...
   ✅ Extracted 121 frames

   🤖 Loading models...
      Loading accident...
      Loading fight_small...
      Loading fire_s...
      Loading nsfw_seg...
      Loading weapon...
      Loading objects...
   ✅ Loaded 6 models: ['accident', 'fight_small', 'fire_s', 'nsfw_seg', 'weapon', 'objects']

   🔍 Testing 121 frames with 6 models...

   📸 Frame 1/121 (Second 1): frame_0001.jpg
      🧠 Testing with accident... ⚪ No detections
      🧠 Testing with fight_small... ⚪ No detections
      🧠 Testing with fire_s... ⚪ No detections
      🧠 Testing with nsfw_seg... ⚪ No detections
      🧠 Testing with weapon... ✅ DETECTED! (1 objects, 0 classes)
      🧠 Testing with objects... ✅ DETECTED! (3 objects, 0 classes)

   📸 Frame 2/121 (Second 2): frame_0002.jpg
      ...
   ```

4. **Review results** in the generated files:
   - `video_name_timestamp_detailed_results.json` - Complete frame-by-frame results
   - `video_name_timestamp_summary.json` - Analysis summary with safety assessment

## Output Structure

### Console Output
- Real-time progress with frame-by-frame analysis
- Model-by-model detection results for each frame
- Final summary with safety assessment and timeline

### Generated Files
1. **Detailed Results JSON**: Complete data for every frame and model
2. **Summary JSON**: High-level analysis with safety assessment
3. **Temporary Frames**: Auto-deleted unless `--keep-frames` is used

### Sample Summary Output
```
============================================================
📊 VIDEO ANALYSIS SUMMARY
============================================================
🎥 Video: test_video.mp4
⏱️  Duration: 120.50s
🔍 Frames Analyzed: 121
⚡ Processing Time: 45.23s
🛡️  Safety Assessment: UNSAFE

📈 DETECTIONS BY MODEL:
  🤖 weapon:
     Frames with detections: 15
     Total detections: 18
     Max confidence: 0.892
     Classes detected: gun, knife

  🤖 objects:
     Frames with detections: 89
     Total detections: 267
     Max confidence: 0.954
     Classes detected: person, car, bottle

⏰ TIMELINE (seconds with detections):
  Second 5: 2 detections
    - weapon: gun (0.856)
    - objects: person (0.923)
  Second 12: 4 detections
    - weapon: knife (0.734)
    - objects: person (0.891)
    - objects: car (0.678)
    ... and 1 more
============================================================
```

## Model Categories

The processor tests each frame with these model categories:

- **🚗 Accident Detection**: Detects vehicle accidents and crashes
- **👊 Fight Detection**: Identifies violence and fighting (nano & small models)
- **🔥 Fire Detection**: Detects fire, smoke, and flames (nano & small models)
- **🔞 NSFW Detection**: Identifies inappropriate content (classification & segmentation)
- **🔫 Weapon Detection**: Detects weapons, guns, knives, etc.
- **📦 Object Detection**: General object detection (COCO dataset)

## Safety Assessment

The system provides an overall safety assessment:
- **SAFE**: No unsafe content detected
- **UNSAFE**: Detected content from unsafe categories (weapon, fight, nsfw, fire, accident)

## Performance Tips

1. **Video Length**: Processing time scales with video duration (1 frame per second)
2. **Model Loading**: Models are loaded once and reused for all frames
3. **Confidence Threshold**: Higher thresholds (0.5-0.7) reduce false positives
4. **Temp Directory**: Use SSD storage for faster frame extraction
5. **GPU Acceleration**: Models will automatically use GPU if available

## Troubleshooting

### FFmpeg Not Found
```
❌ FFmpeg not found! Please install FFmpeg and add it to your PATH.
```
**Solution**: Install FFmpeg and ensure it's in your system PATH

### Model Weights Missing
```
⚠️  Skipping accident - weights not found: d:\akhil\models\accident\weights\yolov8s.pt
```
**Solution**: Ensure model weights are in the correct locations as per the model registry

### Video Format Issues
**Supported formats**: MP4, AVI, MOV, MKV, WMV, FLV, WebM
**Solution**: Convert unsupported formats using FFmpeg

### Memory Issues
For very long videos, consider:
- Processing shorter segments
- Using a higher confidence threshold
- Ensuring sufficient RAM/disk space

## Integration with Existing System

The processor integrates seamlessly with your existing YOLO runner:
- Uses the same model registry and weights
- Compatible with all existing models
- Maintains the same confidence and filtering logic
- Outputs can be used with existing analysis tools

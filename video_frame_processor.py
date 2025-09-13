#!/usr/bin/env python
"""
Video Frame Processor with FFmpeg Integration
Extracts frames from video at 1 second intervals and tests them with all available models.
Automatically cleans up extracted frames after processing.
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile
import argparse

# Add the integrated_yolo_runner to path for imports
sys.path.append(str(Path(__file__).parent / "integrated_yolo_runner"))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics==8.3.58")
    sys.exit(1)

WORKSPACE = Path("d:/akhil").resolve()
MODELS_DIR = WORKSPACE / "models"
INTEGRATED_RUNNER_DIR = WORKSPACE / "integrated_yolo_runner"

# Import model registry from the existing runner
MODEL_REGISTRY = {
    "accident": MODELS_DIR / "accident/weights/yolov8s.pt",
    "fight_nano": MODELS_DIR / "fight/weights/nano_weights.pt", 
    "fight_small": MODELS_DIR / "fight/weights/small_weights.pt",
    "weapon": MODELS_DIR / "weapon/weights/weapon_detection.pt",
    "fire_n": MODELS_DIR / "fire/weights/yolov8n.pt",
    "fire_s": MODELS_DIR / "fire/weights/yolov8s.pt",
    "nsfw_cls": MODELS_DIR / "nsfw/weights/classification_model.pt",
    "nsfw_seg": MODELS_DIR / "nsfw/weights/segmentation_model.pt",
    "objects": Path("yolov8n.pt"),  # Auto-download
}

class VideoFrameProcessor:
    def __init__(self, video_path, output_dir=None, temp_dir=None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else WORKSPACE / "video_analysis_results"
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp(prefix="video_frames_"))
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Video info
        self.video_info = None
        self.total_frames = 0
        self.fps = 0
        self.duration = 0
        
        print(f"🎥 Video: {self.video_path}")
        print(f"📁 Output: {self.output_dir}")
        print(f"🗂️  Temp frames: {self.temp_dir}")
        
    def get_video_info(self):
        """Get video information using FFprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
                str(self.video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
                    
            if video_stream:
                self.fps = eval(video_stream.get('r_frame_rate', '30/1'))
                self.duration = float(video_stream.get('duration', 0))
                self.total_frames = int(video_stream.get('nb_frames', 0))
                
                self.video_info = {
                    'fps': self.fps,
                    'duration': self.duration,
                    'total_frames': self.total_frames,
                    'width': video_stream.get('width'),
                    'height': video_stream.get('height'),
                    'codec': video_stream.get('codec_name')
                }
                
                print(f"📊 Video Info: {self.fps:.2f} FPS, {self.duration:.2f}s, {self.total_frames} frames")
                return True
        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return False
            
    def extract_frames_per_second(self):
        """Extract one frame per second using FFmpeg"""
        if not self.get_video_info():
            return False
            
        print(f"🔄 Extracting frames (1 per second)...")
        
        try:
            # Extract one frame per second
            cmd = [
                'ffmpeg', '-i', str(self.video_path),
                '-vf', 'fps=1',  # 1 frame per second
                '-y',  # Overwrite existing files
                str(self.temp_dir / 'frame_%04d.jpg')
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Count extracted frames
            extracted_frames = list(self.temp_dir.glob('frame_*.jpg'))
            print(f"✅ Extracted {len(extracted_frames)} frames")
            
            return extracted_frames
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
            print(f"stderr: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ Error extracting frames: {e}")
            return False
    
    def load_models(self):
        """Load all available models"""
        models = {}
        print(f"🤖 Loading models...")
        
        for model_name, model_path in MODEL_REGISTRY.items():
            try:
                if model_path.exists() or model_name == "objects":
                    print(f"   Loading {model_name}...")
                    models[model_name] = YOLO(str(model_path))
                else:
                    print(f"   ⚠️  Skipping {model_name} - weights not found: {model_path}")
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {e}")
                
        print(f"✅ Loaded {len(models)} models: {list(models.keys())}")
        return models
    
    def test_frame_with_model(self, frame_path, model_name, model, conf_threshold=0.25):
        """Test a single frame with a specific model"""
        try:
            results = model.predict(source=str(frame_path), conf=conf_threshold, verbose=False)
            
            frame_results = {
                'model': model_name,
                'frame': frame_path.name,
                'detections': [],
                'classifications': [],
                'detected': False
            }
            
            for r in results:
                # Handle detections (bounding boxes)
                if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls.item()) if hasattr(box, 'cls') else -1
                        conf = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
                        xyxy = box.xyxy.squeeze().tolist() if hasattr(box, 'xyxy') else None
                        cls_name = r.names.get(cls_id) if hasattr(r, 'names') and cls_id >= 0 else f"class_{cls_id}"
                        
                        frame_results['detections'].append({
                            'class_id': cls_id,
                            'class_name': cls_name,
                            'confidence': conf,
                            'bbox': xyxy
                        })
                        frame_results['detected'] = True
                
                # Handle classifications
                if hasattr(r, 'probs') and r.probs is not None:
                    try:
                        probs = r.probs.data.tolist()
                        names = r.names if hasattr(r, 'names') else {}
                        
                        for i, prob in enumerate(probs):
                            if prob > conf_threshold:
                                class_name = names.get(i, f"class_{i}")
                                frame_results['classifications'].append({
                                    'class_id': i,
                                    'class_name': class_name,
                                    'probability': prob
                                })
                                frame_results['detected'] = True
                    except Exception as e:
                        print(f"   ⚠️  Classification error for {model_name}: {e}")
            
            return frame_results
            
        except Exception as e:
            print(f"   ❌ Error testing {frame_path.name} with {model_name}: {e}")
            return None
    
    def process_video(self, conf_threshold=0.25):
        """Main processing function"""
        start_time = time.time()
        
        print(f"\n🚀 Starting video processing...")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Extract frames
        extracted_frames = self.extract_frames_per_second()
        if not extracted_frames:
            return False
        
        # Load models
        models = self.load_models()
        if not models:
            print("❌ No models loaded!")
            return False
        
        # Process each frame with each model
        all_results = {
            'video_path': str(self.video_path),
            'video_info': self.video_info,
            'processing_time': None,
            'total_frames_processed': len(extracted_frames),
            'models_used': list(models.keys()),
            'confidence_threshold': conf_threshold,
            'frame_results': []
        }
        
        print(f"\n🔍 Testing {len(extracted_frames)} frames with {len(models)} models...")
        
        for i, frame_path in enumerate(sorted(extracted_frames), 1):
            frame_second = i  # Since we extract 1 frame per second
            print(f"\n📸 Frame {i}/{len(extracted_frames)} (Second {frame_second}): {frame_path.name}")
            
            frame_data = {
                'frame_number': i,
                'second': frame_second,
                'frame_file': frame_path.name,
                'model_results': {}
            }
            
            # Test with each model
            for model_name, model in models.items():
                print(f"   🧠 Testing with {model_name}...", end=" ")
                
                result = self.test_frame_with_model(frame_path, model_name, model, conf_threshold)
                if result:
                    frame_data['model_results'][model_name] = result
                    
                    if result['detected']:
                        det_count = len(result['detections'])
                        cls_count = len(result['classifications'])
                        print(f"✅ DETECTED! ({det_count} objects, {cls_count} classes)")
                    else:
                        print("⚪ No detections")
                else:
                    print("❌ Error")
            
            all_results['frame_results'].append(frame_data)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        all_results['processing_time'] = processing_time
        
        # Save results
        self.save_results(all_results)
        
        # Generate summary
        self.generate_summary(all_results)
        
        print(f"\n✅ Processing complete! Time: {processing_time:.2f}s")
        return all_results
    
    def save_results(self, results):
        """Save detailed results to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = self.video_path.stem
        
        results_file = self.output_dir / f"{video_name}_{timestamp}_detailed_results.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Detailed results saved: {results_file}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def generate_summary(self, results):
        """Generate and save a summary report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = self.video_path.stem
        
        # Analyze results
        summary = {
            'video_info': {
                'file': str(self.video_path),
                'duration': results['video_info']['duration'],
                'fps': results['video_info']['fps'],
                'total_frames': results['video_info']['total_frames']
            },
            'analysis_summary': {
                'frames_analyzed': results['total_frames_processed'],
                'models_used': results['models_used'],
                'processing_time': results['processing_time']
            },
            'detections_by_model': {},
            'detections_by_second': {},
            'overall_safety_assessment': 'SAFE'
        }
        
        # Count detections by model
        for model_name in results['models_used']:
            summary['detections_by_model'][model_name] = {
                'frames_with_detections': 0,
                'total_detections': 0,
                'max_confidence': 0.0,
                'detected_classes': set()
            }
        
        # Analyze each frame
        unsafe_categories = {'weapon', 'fight_nano', 'fight_small', 'nsfw_cls', 'nsfw_seg', 'fire_n', 'fire_s', 'accident'}
        
        for frame_data in results['frame_results']:
            second = frame_data['second']
            summary['detections_by_second'][second] = []
            
            for model_name, model_result in frame_data['model_results'].items():
                if model_result['detected']:
                    summary['detections_by_model'][model_name]['frames_with_detections'] += 1
                    
                    # Count detections
                    for det in model_result['detections']:
                        summary['detections_by_model'][model_name]['total_detections'] += 1
                        summary['detections_by_model'][model_name]['max_confidence'] = max(
                            summary['detections_by_model'][model_name]['max_confidence'],
                            det['confidence']
                        )
                        summary['detections_by_model'][model_name]['detected_classes'].add(det['class_name'])
                        
                        # Add to second summary
                        summary['detections_by_second'][second].append({
                            'model': model_name,
                            'class': det['class_name'],
                            'confidence': det['confidence']
                        })
                        
                        # Check if unsafe
                        if model_name in unsafe_categories:
                            summary['overall_safety_assessment'] = 'UNSAFE'
                    
                    # Count classifications
                    for cls in model_result['classifications']:
                        summary['detections_by_model'][model_name]['detected_classes'].add(cls['class_name'])
                        summary['detections_by_second'][second].append({
                            'model': model_name,
                            'class': cls['class_name'],
                            'probability': cls['probability']
                        })
                        
                        if model_name in unsafe_categories:
                            summary['overall_safety_assessment'] = 'UNSAFE'
        
        # Convert sets to lists for JSON serialization
        for model_data in summary['detections_by_model'].values():
            model_data['detected_classes'] = list(model_data['detected_classes'])
        
        # Save summary
        summary_file = self.output_dir / f"{video_name}_{timestamp}_summary.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"📊 Summary saved: {summary_file}")
        except Exception as e:
            print(f"❌ Error saving summary: {e}")
        
        # Print summary to console
        self.print_summary(summary)
    
    def print_summary(self, summary):
        """Print a formatted summary to console"""
        print(f"\n{'='*60}")
        print(f"📊 VIDEO ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"🎥 Video: {Path(summary['video_info']['file']).name}")
        print(f"⏱️  Duration: {summary['video_info']['duration']:.2f}s")
        print(f"🔍 Frames Analyzed: {summary['analysis_summary']['frames_analyzed']}")
        print(f"⚡ Processing Time: {summary['analysis_summary']['processing_time']:.2f}s")
        print(f"🛡️  Safety Assessment: {summary['overall_safety_assessment']}")
        
        print(f"\n📈 DETECTIONS BY MODEL:")
        for model_name, data in summary['detections_by_model'].items():
            if data['total_detections'] > 0:
                print(f"  🤖 {model_name}:")
                print(f"     Frames with detections: {data['frames_with_detections']}")
                print(f"     Total detections: {data['total_detections']}")
                print(f"     Max confidence: {data['max_confidence']:.3f}")
                print(f"     Classes detected: {', '.join(data['detected_classes'])}")
        
        print(f"\n⏰ TIMELINE (seconds with detections):")
        for second, detections in summary['detections_by_second'].items():
            if detections:
                print(f"  Second {second}: {len(detections)} detections")
                for det in detections[:3]:  # Show first 3
                    conf_key = 'confidence' if 'confidence' in det else 'probability'
                    print(f"    - {det['model']}: {det['class']} ({det[conf_key]:.3f})")
                if len(detections) > 3:
                    print(f"    ... and {len(detections) - 3} more")
        
        print(f"{'='*60}")
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temporary files: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp files: {e}")

def main():
    parser = argparse.ArgumentParser(description="Video Frame Processor with Multi-Model Testing")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--temp", "-t", help="Temporary directory for extracted frames")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames after processing")
    
    args = parser.parse_args()
    
    if not Path(args.video_path).exists():
        print(f"❌ Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found! Please install FFmpeg and add it to your PATH.")
        sys.exit(1)
    
    processor = VideoFrameProcessor(
        video_path=args.video_path,
        output_dir=args.output,
        temp_dir=args.temp
    )
    
    try:
        results = processor.process_video(conf_threshold=args.conf)
        
        if results:
            print(f"\n🎉 Video processing completed successfully!")
            print(f"📁 Results saved to: {processor.output_dir}")
        else:
            print(f"\n❌ Video processing failed!")
            
    finally:
        if not args.keep_frames:
            processor.cleanup()

if __name__ == "__main__":
    main()

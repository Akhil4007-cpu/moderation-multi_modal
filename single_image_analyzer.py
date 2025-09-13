#!/usr/bin/env python
"""
Single Image Analyzer - Test a single image with all available models
Shows clear detection results for weapon, accident, fight, fire, nsfw, and objects
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Add the integrated_yolo_runner to path
sys.path.append(str(Path(__file__).parent / "integrated_yolo_runner"))

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: pip install ultralytics opencv-python")
    sys.exit(1)

class SingleImageAnalyzer:
    def __init__(self):
        """Initialize the single image analyzer with all available models"""
        self.models = {}
        self.model_info = {}
        self.load_models()
    
    def load_models(self):
        """Load all available YOLO models"""
        
        # Model registry - based on actual model file structure
        model_registry = {
            'weapon': {
                'path': 'models/weapon/weights/weapon_detection.pt',
                'task': 'detect',
                'categories': ['weapon']
            },
            'accident': {
                'path': 'models/accident/weights/yolov8s.pt', 
                'task': 'detect',
                'categories': ['accident']
            },
            'fight_nano': {
                'path': 'models/fight/weights/nano_weights.pt',
                'task': 'detect', 
                'categories': ['fight']
            },
            'fight_small': {
                'path': 'models/fight/weights/small_weights.pt',
                'task': 'detect',
                'categories': ['fight']
            },
            'fire_n': {
                'path': 'models/fire/weights/yolov8n.pt',
                'task': 'detect',
                'categories': ['fire']
            },
            'fire_s': {
                'path': 'models/fire/weights/yolov8s.pt', 
                'task': 'detect',
                'categories': ['fire']
            },
            'nsfw_cls': {
                'path': 'models/nsfw/weights/classification_model.pt',
                'task': 'classify',
                'categories': ['nsfw']
            },
            'nsfw_seg': {
                'path': 'models/nsfw/weights/segmentation_model.pt',
                'task': 'segment',
                'categories': ['nsfw']
            },
            'objects': {
                'path': 'yolov8n.pt',
                'task': 'detect',
                'categories': ['objects']
            }
        }
        
        print("🔄 Loading models...")
        
        for model_name, model_config in model_registry.items():
            model_path = Path(model_config['path'])
            
            # Try absolute path first, then relative to script directory
            if not model_path.exists():
                model_path = Path(__file__).parent / model_config['path']
            
            if model_path.exists():
                try:
                    print(f"   Loading {model_name}...")
                    model = YOLO(str(model_path))
                    self.models[model_name] = model
                    self.model_info[model_name] = {
                        'path': str(model_path),
                        'task': model_config['task'],
                        'categories': model_config['categories']
                    }
                    print(f"   ✅ {model_name} loaded successfully")
                except Exception as e:
                    print(f"   ❌ Failed to load {model_name}: {e}")
            else:
                print(f"   ⚠️  Model not found: {model_path}")
        
        print(f"✅ Loaded {len(self.models)} models successfully\n")
    
    def analyze_image(self, image_path, conf_threshold=0.25):
        """Analyze a single image with all loaded models"""
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"🔍 Analyzing image: {image_path.name}")
        print(f"📁 Full path: {image_path}")
        
        # Load image info
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Could not load image")
            height, width = img.shape[:2]
            file_size = image_path.stat().st_size
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
        
        # Initialize results structure
        results = {
            'image_path': str(image_path),
            'image_info': {
                'filename': image_path.name,
                'width': width,
                'height': height,
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024*1024), 2)
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'confidence_threshold': conf_threshold,
            'models_used': list(self.models.keys()),
            'model_results': {},
            'summary': {
                'total_detections': 0,
                'total_classifications': 0,
                'models_with_detections': 0,
                'safety_assessment': 'SAFE',
                'categories_detected': []
            }
        }
        
        print(f"📏 Image dimensions: {width}x{height}")
        print(f"💾 File size: {results['image_info']['file_size_mb']} MB")
        print(f"🎯 Confidence threshold: {conf_threshold}")
        print(f"🤖 Testing with {len(self.models)} models...\n")
        
        # Test each model
        for model_name, model in self.models.items():
            print(f"🔄 Testing {model_name}...")
            
            model_result = {
                'model_name': model_name,
                'model_info': self.model_info[model_name],
                'detected': False,
                'detections': [],
                'classifications': [],
                'processing_time_ms': 0
            }
            
            try:
                # Run prediction
                start_time = datetime.now()
                predictions = model(str(image_path), conf=conf_threshold, verbose=False)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds() * 1000
                model_result['processing_time_ms'] = round(processing_time, 2)
                
                # Process results based on task type
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    
                    if self.model_info[model_name]['task'] in ['detect', 'segment']:
                        # Detection/Segmentation results
                        if hasattr(pred, 'boxes') and pred.boxes is not None and len(pred.boxes) > 0:
                            model_result['detected'] = True
                            
                            for box in pred.boxes:
                                detection = {
                                    'class_id': int(box.cls.item()),
                                    'class_name': pred.names[int(box.cls.item())],
                                    'confidence': float(box.conf.item()),
                                    'bbox': [float(x) for x in box.xyxy[0].tolist()],  # [x1, y1, x2, y2]
                                    'bbox_center': [
                                        float((box.xyxy[0][0] + box.xyxy[0][2]) / 2),
                                        float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                                    ],
                                    'bbox_size': [
                                        float(box.xyxy[0][2] - box.xyxy[0][0]),
                                        float(box.xyxy[0][3] - box.xyxy[0][1])
                                    ]
                                }
                                model_result['detections'].append(detection)
                                results['summary']['total_detections'] += 1
                    
                    elif self.model_info[model_name]['task'] == 'classify':
                        # Classification results
                        if hasattr(pred, 'probs') and pred.probs is not None:
                            model_result['detected'] = True
                            
                            # Get top predictions
                            top_indices = pred.probs.top5
                            top_confidences = pred.probs.top5conf
                            
                            for idx, conf in zip(top_indices, top_confidences):
                                classification = {
                                    'class_id': int(idx),
                                    'class_name': pred.names[int(idx)],
                                    'probability': float(conf)
                                }
                                model_result['classifications'].append(classification)
                                results['summary']['total_classifications'] += 1
                
                # Update summary
                if model_result['detected']:
                    results['summary']['models_with_detections'] += 1
                    results['summary']['categories_detected'].extend(self.model_info[model_name]['categories'])
                    
                    # Check for unsafe categories
                    unsafe_categories = {'weapon', 'fight', 'fire', 'nsfw', 'accident'}
                    if any(cat in unsafe_categories for cat in self.model_info[model_name]['categories']):
                        results['summary']['safety_assessment'] = 'UNSAFE'
                
                print(f"   ✅ {model_name}: {len(model_result['detections'])} detections, {len(model_result['classifications'])} classifications ({processing_time:.1f}ms)")
                
            except Exception as e:
                print(f"   ❌ {model_name} failed: {e}")
                model_result['error'] = str(e)
            
            results['model_results'][model_name] = model_result
        
        # Remove duplicates from categories
        results['summary']['categories_detected'] = list(set(results['summary']['categories_detected']))
        
        return results
    
    def print_results(self, results):
        """Print formatted analysis results"""
        
        if not results:
            print("❌ No results to display")
            return
        
        print(f"\n{'='*80}")
        print(f"🎯 SINGLE IMAGE ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        # Image info
        img_info = results['image_info']
        print(f"🖼️  Image: {img_info['filename']}")
        print(f"📏 Dimensions: {img_info['width']}x{img_info['height']}")
        print(f"💾 Size: {img_info['file_size_mb']} MB")
        print(f"🎯 Confidence Threshold: {results['confidence_threshold']}")
        
        # Summary
        summary = results['summary']
        safety_icon = "🚨" if summary['safety_assessment'] == 'UNSAFE' else "✅"
        print(f"\n📊 SUMMARY:")
        print(f"   Total Detections: {summary['total_detections']}")
        print(f"   Total Classifications: {summary['total_classifications']}")
        print(f"   Models with Results: {summary['models_with_detections']}/{len(results['models_used'])}")
        print(f"   Categories Found: {', '.join(summary['categories_detected']) if summary['categories_detected'] else 'None'}")
        print(f"   {safety_icon} Safety Assessment: {summary['safety_assessment']}")
        
        # Detailed results by category
        categories = ['weapon', 'accident', 'fight_nano', 'fight_small', 'fire_n', 'fire_s', 'nsfw_cls', 'nsfw_seg', 'objects']
        category_icons = {
            'weapon': '🔫', 'accident': '🚗', 'fight_nano': '👊', 'fight_small': '🥊',
            'fire_n': '🔥', 'fire_s': '🔥', 'nsfw_cls': '🔞', 'nsfw_seg': '🔞', 'objects': '📦'
        }
        
        for category in categories:
            if category in results['model_results']:
                model_result = results['model_results'][category]
                icon = category_icons.get(category, '🔍')
                
                print(f"\n{'-'*60}")
                print(f"{icon} {category.upper()} ANALYSIS")
                print(f"{'-'*60}")
                
                if model_result['detected']:
                    print(f"✅ Status: DETECTED")
                    print(f"⏱️  Processing Time: {model_result['processing_time_ms']}ms")
                    
                    # Show detections
                    if model_result['detections']:
                        print(f"\n🎯 DETECTIONS ({len(model_result['detections'])}):")
                        for i, det in enumerate(model_result['detections'], 1):
                            conf_icon = "🔴" if det['confidence'] >= 0.7 else "🟡" if det['confidence'] >= 0.5 else "🟢"
                            print(f"   {i}. {conf_icon} {det['class_name']}")
                            print(f"      Confidence: {det['confidence']:.3f}")
                            print(f"      Position: ({det['bbox_center'][0]:.0f}, {det['bbox_center'][1]:.0f})")
                            print(f"      Size: {det['bbox_size'][0]:.0f}x{det['bbox_size'][1]:.0f}")
                            print(f"      Bounding Box: [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")
                    
                    # Show classifications
                    if model_result['classifications']:
                        print(f"\n📋 CLASSIFICATIONS ({len(model_result['classifications'])}):")
                        for i, cls in enumerate(model_result['classifications'], 1):
                            prob_icon = "🔴" if cls['probability'] >= 0.7 else "🟡" if cls['probability'] >= 0.5 else "🟢"
                            print(f"   {i}. {prob_icon} {cls['class_name']}")
                            print(f"      Probability: {cls['probability']:.3f}")
                else:
                    print(f"❌ Status: NOT DETECTED")
                    print(f"⏱️  Processing Time: {model_result['processing_time_ms']}ms")
                    
                    if 'error' in model_result:
                        print(f"⚠️  Error: {model_result['error']}")
        
        print(f"\n{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Single Image Analyzer with Multi-Model Testing")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output", "-o", help="Output JSON file path (optional)")
    parser.add_argument("--no-display", action="store_true", help="Don't display results, only save to file")
    
    args = parser.parse_args()
    
    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Initialize analyzer
    try:
        analyzer = SingleImageAnalyzer()
        
        if len(analyzer.models) == 0:
            print("❌ No models loaded. Please check model paths.")
            sys.exit(1)
        
        # Analyze image
        results = analyzer.analyze_image(image_path, args.conf)
        
        if not results:
            print("❌ Analysis failed")
            sys.exit(1)
        
        # Display results
        if not args.no_display:
            analyzer.print_results(results)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
        else:
            # Auto-generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = image_path.parent / f"{image_path.stem}_analysis_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

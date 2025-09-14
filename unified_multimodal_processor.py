#!/usr/bin/env python
"""
Unified Multi-Modal Content Moderation System
Integrates video, image, audio, and text processing into a single comprehensive system
"""

import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

# Add local modules to path
sys.path.append(str(Path(__file__).parent / "integrated_yolo_runner"))
sys.path.append(str(Path(__file__).parent / "audio"))
sys.path.append(str(Path(__file__).parent / "text"))

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    import librosa
    import joblib
    from pydub import AudioSegment
    import pytesseract
    import speech_recognition as sr
    import re
    import pickle
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: pip install ultralytics opencv-python librosa joblib pydub pytesseract SpeechRecognition scikit-learn")
    sys.exit(1)

class UnifiedMultiModalProcessor:
    def __init__(self):
        """Initialize the unified multi-modal processor"""
        self.models = {}
        self.audio_model = None
        self.text_model = None
        self.load_all_models()
        
        # Unsafe keywords for text analysis
        self.unsafe_keywords = [
            "suicide", "murder", "kill", "drugs", "weapon", "crime",
            "adult", "abuse", "fight", "fuck", "bitch", "bloody", "asshole",
            "violence", "hate", "terrorism", "bomb", "gun", "knife"
        ]
    
    def load_all_models(self):
        """Load all available models (YOLO, Audio, Text)"""
        print("🔄 Loading all models...")
        
        # Load YOLO models
        model_registry = {
            'weapon': 'models/weapon/weights/weapon_detection.pt',
            'accident': 'models/accident/weights/yolov8s.pt',
            'fight_nano': 'models/fight/weights/nano_weights.pt',
            'fight_small': 'models/fight/weights/small_weights.pt',
            'fire_n': 'models/fire/weights/yolov8n.pt',
            'fire_s': 'models/fire/weights/yolov8s.pt',
            'nsfw_cls': 'models/nsfw/weights/classification_model.pt',
            'nsfw_seg': 'models/nsfw/weights/segmentation_model.pt',
            'objects': 'yolov8n.pt'
        }
        
        for model_name, model_path in model_registry.items():
            model_path = Path(model_path)
            if not model_path.exists():
                model_path = Path(__file__).parent / model_path
            
            if model_path.exists():
                try:
                    self.models[model_name] = YOLO(str(model_path))
                    print(f"   ✅ {model_name} loaded")
                except Exception as e:
                    print(f"   ❌ Failed to load {model_name}: {e}")
            else:
                print(f"   ⚠️  Model not found: {model_path}")
        
        # Load audio model
        audio_model_path = Path("audio/audio_classifier_sklearn.pkl")
        if audio_model_path.exists():
            try:
                self.audio_model = joblib.load(audio_model_path)
                print("   ✅ Audio classifier loaded")
            except Exception as e:
                print(f"   ❌ Failed to load audio model: {e}")
        
        # Load text model (try fixed model first, then fallback)
        text_model_paths = [
            Path("text/text_model_fixed.pkl"),
            Path("text/text_model.pkl")
        ]
        
        for text_model_path in text_model_paths:
            if text_model_path.exists():
                try:
                    if 'fixed' in str(text_model_path):
                        self.text_model = joblib.load(text_model_path)
                    else:
                        with open(text_model_path, 'rb') as f:
                            self.text_model = pickle.load(f)
                    print(f"   ✅ Text classifier loaded from {text_model_path.name}")
                    break
                except Exception as e:
                    print(f"   ⚠️  Failed to load {text_model_path.name}: {e}")
                    continue
        
        if not hasattr(self, 'text_model') or self.text_model is None:
            print("   ⚠️  No working text model found, using keyword-based detection")
        
        print(f"✅ Loaded {len(self.models)} YOLO models, audio: {'✅' if self.audio_model else '❌'}, text: {'✅ (ML)' if self.text_model else '✅ (Keywords)'}")
    
    def process_image(self, image_path, conf_threshold=0.5):
        """Process single image with all YOLO models"""
        results = {
            'modality': 'image',
            'file_path': str(image_path),
            'models_tested': list(self.models.keys()),
            'detections': {},
            'summary': {
                'total_detections': 0,
                'unsafe_categories': [],
                'safety_assessment': 'SAFE'
            }
        }
        
        for model_name, model in self.models.items():
            try:
                predictions = model(str(image_path), conf=conf_threshold, verbose=False)
                model_result = {'detected': False, 'detections': [], 'classifications': []}
                
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    
                    # Handle detections
                    if hasattr(pred, 'boxes') and pred.boxes is not None and len(pred.boxes) > 0:
                        model_result['detected'] = True
                        for box in pred.boxes:
                            detection = {
                                'class_name': pred.names[int(box.cls.item())],
                                'confidence': float(box.conf.item()),
                                'bbox': [float(x) for x in box.xyxy[0].tolist()]
                            }
                            model_result['detections'].append(detection)
                            results['summary']['total_detections'] += 1
                    
                    # Handle classifications
                    if hasattr(pred, 'probs') and pred.probs is not None:
                        model_result['detected'] = True
                        top_indices = pred.probs.top5
                        top_confidences = pred.probs.top5conf
                        
                        for idx, conf in zip(top_indices, top_confidences):
                            classification = {
                                'class_name': pred.names[int(idx)],
                                'probability': float(conf)
                            }
                            model_result['classifications'].append(classification)
                
                results['detections'][model_name] = model_result
                
                # Check for unsafe categories
                unsafe_categories = {'weapon', 'fight', 'fire', 'nsfw', 'accident'}
                if model_result['detected'] and any(cat in model_name for cat in unsafe_categories):
                    results['summary']['unsafe_categories'].append(model_name)
                    results['summary']['safety_assessment'] = 'UNSAFE'
                    
            except Exception as e:
                results['detections'][model_name] = {'error': str(e)}
        
        return results
    
    def process_audio(self, audio_path):
        """Process audio file with audio classifier"""
        if not self.audio_model:
            return {'error': 'Audio model not loaded'}
        
        try:
            # Convert video to audio if needed
            original_path = audio_path
            if str(audio_path).lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                temp_audio = "temp_audio_unified.wav"
                AudioSegment.from_file(audio_path).export(temp_audio, format="wav")
                audio_path = temp_audio
            
            # Extract features
            y, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features = np.concatenate([np.mean(mfcc.T, axis=0),
                                     np.mean(chroma.T, axis=0),
                                     np.mean(contrast.T, axis=0)])
            
            # Make prediction
            features = features.reshape(1, -1)
            categories = ["adult", "drugs", "hate", "safe", "spam", "violence"]
            prediction = self.audio_model.predict(features)[0]
            probabilities = self.audio_model.predict_proba(features)[0]
            
            predicted_category = categories[prediction]
            confidence = probabilities[prediction] * 100
            is_safe = predicted_category == "safe"
            
            # Clean up temp file
            if original_path != audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            
            return {
                'modality': 'audio',
                'file_path': str(original_path),
                'predicted_category': predicted_category,
                'confidence': round(confidence, 2),
                'is_safe': is_safe,
                'safety_status': 'SAFE' if is_safe else 'UNSAFE',
                'all_probabilities': {cat: round(prob * 100, 2) for cat, prob in zip(categories, probabilities)}
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def process_text_content(self, text_content):
        """Process text content for unsafe keywords and classification"""
        if not text_content.strip():
            return {'error': 'No text content provided'}
        
        # Find unsafe keywords
        unsafe_words = []
        text_lower = text_content.lower()
        for word in self.unsafe_keywords:
            if re.search(r"\b" + re.escape(word) + r"\b", text_lower):
                unsafe_words.append(word)
        
        # Basic safety assessment
        is_safe = len(unsafe_words) == 0
        confidence = 1.0 if is_safe else 0.99
        
        result = {
            'modality': 'text',
            'text_content': text_content,
            'unsafe_words_found': unsafe_words,
            'is_safe': is_safe,
            'safety_status': 'SAFE' if is_safe else 'UNSAFE',
            'confidence': confidence
        }
        
        # Use ML model if available
        if self.text_model:
            try:
                # Use the ML model for classification
                ml_prediction = self.text_model.predict([text_content])[0]
                ml_probabilities = self.text_model.predict_proba([text_content])[0]
                ml_confidence = max(ml_probabilities) * 100
                
                # ML model: 0 = safe, 1 = unsafe
                ml_is_safe = ml_prediction == 0
                ml_safety_status = 'SAFE' if ml_is_safe else 'UNSAFE'
                
                # Combine ML and keyword results (use stricter assessment)
                combined_is_safe = is_safe and ml_is_safe
                combined_confidence = min(confidence * 100, ml_confidence)
                
                result.update({
                    'ml_classification': {
                        'prediction': ml_safety_status,
                        'confidence': round(ml_confidence, 2),
                        'method': 'sklearn_pipeline'
                    },
                    'combined_assessment': {
                        'is_safe': combined_is_safe,
                        'safety_status': 'SAFE' if combined_is_safe else 'UNSAFE',
                        'confidence': round(combined_confidence, 2)
                    }
                })
                
                # Update main result with combined assessment
                result['is_safe'] = combined_is_safe
                result['safety_status'] = 'SAFE' if combined_is_safe else 'UNSAFE'
                result['confidence'] = round(combined_confidence, 2)
                
            except Exception as e:
                result['ml_error'] = str(e)
                result['classification_method'] = 'keyword_based_fallback'
        else:
            result['classification_method'] = 'keyword_based'
        
        return result
    
    def extract_text_from_video(self, video_path):
        """Extract text from video using OCR and speech recognition"""
        frame_text = ""
        audio_text = ""
        
        try:
            # Extract text from frames (OCR)
            cap = cv2.VideoCapture(str(video_path))
            frame_count = 0
            extracted_texts = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 30 == 0:  # Every 30 frames
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    try:
                        text = pytesseract.image_to_string(gray)
                        if text.strip():
                            extracted_texts.append(text.strip())
                    except:
                        pass
                
                frame_count += 1
            
            cap.release()
            frame_text = " ".join(extracted_texts) if extracted_texts else ""
            
        except Exception as e:
            print(f"⚠️  OCR extraction failed: {e}")
        
        try:
            # Extract audio and convert to text (ASR)
            temp_audio = "temp_audio_text.wav"
            command = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_audio}"'
            subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(temp_audio) as source:
                    audio = recognizer.record(source)
                    audio_text = recognizer.recognize_google(audio)
            except:
                pass
            
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
                
        except Exception as e:
            print(f"⚠️  ASR extraction failed: {e}")
        
        return (frame_text + " " + audio_text).strip()
    
    def process_video_comprehensive(self, video_path, conf_threshold=0.5):
        """Comprehensive video processing: frames + audio + text"""
        print(f"🎥 Processing video comprehensively: {Path(video_path).name}")
        
        results = {
            'modality': 'video_comprehensive',
            'file_path': str(video_path),
            'timestamp': datetime.now().isoformat(),
            'video_analysis': None,
            'audio_analysis': None,
            'text_analysis': None,
            'unified_assessment': {
                'overall_safety': 'SAFE',
                'confidence_score': 0.0,
                'unsafe_modalities': [],
                'summary': ''
            }
        }
        
        # 1. Video frame analysis (using existing video processor logic)
        print("🔍 Analyzing video frames...")
        try:
            # Use simplified frame extraction for demo
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps) if fps > 0 else 30
            
            frame_results = []
            frame_count = 0
            
            while frame_count < 5:  # Analyze first 5 seconds for demo
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Save frame temporarily and analyze
                    temp_frame = f"temp_frame_{frame_count}.jpg"
                    cv2.imwrite(temp_frame, frame)
                    
                    frame_analysis = self.process_image(temp_frame, conf_threshold)
                    frame_results.append(frame_analysis)
                    
                    os.remove(temp_frame)
                
                frame_count += 1
            
            cap.release()
            results['video_analysis'] = {
                'frames_analyzed': len(frame_results),
                'frame_results': frame_results
            }
            
        except Exception as e:
            results['video_analysis'] = {'error': str(e)}
        
        # 2. Audio analysis
        print("🔊 Analyzing audio content...")
        results['audio_analysis'] = self.process_audio(video_path)
        
        # 3. Text extraction and analysis
        print("📝 Extracting and analyzing text...")
        extracted_text = self.extract_text_from_video(video_path)
        results['text_analysis'] = self.process_text_content(extracted_text)
        
        # 4. Unified assessment
        unsafe_modalities = []
        confidence_scores = []
        
        # Check video safety
        if results['video_analysis'] and 'frame_results' in results['video_analysis']:
            for frame_result in results['video_analysis']['frame_results']:
                if frame_result.get('summary', {}).get('safety_assessment') == 'UNSAFE':
                    unsafe_modalities.append('video')
                    break
        
        # Check audio safety
        if results['audio_analysis'] and not results['audio_analysis'].get('error'):
            if results['audio_analysis'].get('safety_status') == 'UNSAFE':
                unsafe_modalities.append('audio')
            confidence_scores.append(results['audio_analysis'].get('confidence', 0))
        
        # Check text safety
        if results['text_analysis'] and not results['text_analysis'].get('error'):
            if results['text_analysis'].get('safety_status') == 'UNSAFE':
                unsafe_modalities.append('text')
            confidence_scores.append(results['text_analysis'].get('confidence', 0) * 100)
        
        # Overall assessment
        overall_safety = 'UNSAFE' if unsafe_modalities else 'SAFE'
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        results['unified_assessment'] = {
            'overall_safety': overall_safety,
            'confidence_score': round(avg_confidence, 2),
            'unsafe_modalities': unsafe_modalities,
            'summary': f"Analysis complete. {'⚠️ UNSAFE' if overall_safety == 'UNSAFE' else '✅ SAFE'} content detected across {len(unsafe_modalities)} modalities." if unsafe_modalities else "✅ Content appears safe across all modalities."
        }
        
        return results
    
    def process_file(self, file_path, mode='auto', conf_threshold=0.5):
        """Process any file based on its type or specified mode"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': f'File not found: {file_path}'}
        
        # Determine processing mode
        if mode == 'auto':
            ext = file_path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                mode = 'image'
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                mode = 'video'
            elif ext in ['.mp3', '.wav', '.flac', '.m4a']:
                mode = 'audio'
            elif ext in ['.txt', '.md']:
                mode = 'text'
            else:
                return {'error': f'Unsupported file type: {ext}'}
        
        # Process based on mode
        if mode == 'image':
            return self.process_image(file_path, conf_threshold)
        elif mode == 'video':
            return self.process_video_comprehensive(file_path, conf_threshold)
        elif mode == 'video_comprehensive':
            return self.process_video_comprehensive(file_path, conf_threshold)
        elif mode == 'audio':
            return self.process_audio(file_path)
        elif mode == 'text':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.process_text_content(content)
        else:
            return {'error': f'Invalid processing mode: {mode}'}

def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Modal Content Moderation System")
    parser.add_argument("file_path", help="Path to the file to process")
    parser.add_argument("--mode", choices=['auto', 'image', 'video', 'video_comprehensive', 'audio', 'text'], 
                       default='auto', help="Processing mode (default: auto-detect)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--no-display", action="store_true", help="Don't display results, only save")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = UnifiedMultiModalProcessor()
    
    # Process file
    print(f"🚀 Starting unified multi-modal processing...")
    results = processor.process_file(args.file_path, args.mode, args.conf)
    
    # Display results
    if not args.no_display:
        print("\n" + "="*80)
        print("🎯 UNIFIED MULTI-MODAL ANALYSIS RESULTS")
        print("="*80)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"📁 File: {Path(args.file_path).name}")
            print(f"🔍 Modality: {results.get('modality', 'unknown')}")
            
            if results.get('modality') == 'video_comprehensive':
                assessment = results.get('unified_assessment', {})
                print(f"🛡️  Overall Safety: {assessment.get('overall_safety', 'UNKNOWN')}")
                print(f"📊 Confidence: {assessment.get('confidence_score', 0)}%")
                print(f"⚠️  Unsafe Modalities: {', '.join(assessment.get('unsafe_modalities', [])) or 'None'}")
                print(f"📝 Summary: {assessment.get('summary', 'No summary available')}")
            else:
                safety = results.get('safety_status') or results.get('summary', {}).get('safety_assessment', 'UNKNOWN')
                print(f"🛡️  Safety Status: {safety}")
                
                if 'confidence' in results:
                    print(f"📊 Confidence: {results['confidence']}%")
        
        print("="*80)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(args.file_path).parent / f"{Path(args.file_path).stem}_unified_analysis_{timestamp}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()

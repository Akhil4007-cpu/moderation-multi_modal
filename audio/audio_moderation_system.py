import os
import sys
import librosa
import numpy as np
import joblib
import json
from datetime import datetime
from pydub import AudioSegment

# ---------------- CONFIG ----------------
MODEL_PATH = "audio_classifier_sklearn.pkl"
CATEGORIES = ["adult", "drugs", "hate", "safe", "spam", "violence"]
UNSAFE = ["adult", "drugs", "hate", "spam", "violence"]

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(file_path):
    """Extract audio features for classification"""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features = np.concatenate([np.mean(mfcc.T, axis=0),
                                   np.mean(chroma.T, axis=0),
                                   np.mean(contrast.T, axis=0)])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# ---------------- LOAD MODEL ----------------
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# ---------------- MODERATION FUNCTION ----------------
def moderate_audio(file_path, model):
    """Moderate audio/video file and return results"""
    try:
        # Convert video to audio if needed
        original_path = file_path
        if file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print("🎬 Converting video to audio...")
            audio_path = "temp_audio.wav"
            AudioSegment.from_file(file_path).export(audio_path, format="wav")
            file_path = audio_path
        
        # Extract features
        print("🔍 Extracting audio features...")
        features = extract_features(file_path)
        if features is None:
            return None
        
        # Make prediction
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get results
        predicted_category = CATEGORIES[prediction]
        confidence = probabilities[prediction] * 100
        is_safe = predicted_category == "safe"
        
        # Create detailed results
        all_probabilities = {}
        for i, cat in enumerate(CATEGORIES):
            all_probabilities[cat] = round(probabilities[i] * 100, 2)
        
        # Clean up temporary file
        if original_path != file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "file_path": original_path,
            "file_name": os.path.basename(original_path),
            "predicted_category": predicted_category,
            "confidence": round(confidence, 2),
            "is_safe": is_safe,
            "safety_status": "SAFE" if is_safe else "UNSAFE",
            "all_probabilities": all_probabilities,
            "timestamp": datetime.now().isoformat(),
            "model_used": MODEL_PATH
        }
        
    except Exception as e:
        print(f"❌ Error during moderation: {e}")
        return None

# ---------------- SAVE RESULTS ----------------
def save_results(results, output_file="moderation_results.json"):
    """Save moderation results to JSON file"""
    try:
        # Load existing results if file exists
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
        else:
            existing_results = {"moderation_history": []}
        
        # Add new result
        existing_results["moderation_history"].append(results)
        existing_results["last_updated"] = datetime.now().isoformat()
        existing_results["total_moderations"] = len(existing_results["moderation_history"])
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        print(f"💾 Results saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

# ---------------- DISPLAY RESULTS ----------------
def display_results(results):
    """Display moderation results in a nice format"""
    print("\n" + "="*60)
    print("🎵 AUDIO MODERATION RESULTS")
    print("="*60)
    print(f"📁 File: {results['file_name']}")
    print(f"🏷️  Category: {results['predicted_category']}")
    print(f"📊 Confidence: {results['confidence']}%")
    print(f"🛡️  Safety Status: {results['safety_status']}")
    print(f"⏰ Timestamp: {results['timestamp']}")
    
    print(f"\n📈 All Category Probabilities:")
    for category, prob in results['all_probabilities'].items():
        status_icon = "✅" if category == results['predicted_category'] else "  "
        print(f"  {status_icon} {category}: {prob}%")
    
    print("="*60)

# ---------------- MAIN FUNCTION ----------------
def main():
    """Main moderation function"""
    if len(sys.argv) < 2:
        print("Usage: python audio_moderation_system.py <audio_or_video_path> [output_json_file]")
        print("Example: python audio_moderation_system.py my_audio.mp3 results.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "moderation_results.json"
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        sys.exit(1)
    
    print(f"🎵 Starting moderation for: {input_path}")
    
    # Load model
    model = load_model()
    if model is None:
        sys.exit(1)
    
    # Moderate the file
    results = moderate_audio(input_path, model)
    if results is None:
        print("❌ Moderation failed")
        sys.exit(1)
    
    # Display results
    display_results(results)
    
    # Save results
    if save_results(results, output_file):
        print(f"✅ Moderation completed successfully!")
    else:
        print("⚠️  Moderation completed but failed to save results")

if __name__ == "__main__":
    main()

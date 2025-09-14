import os
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
TEST_DATASET_PATH = "dataset/test"

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
        print(f"Error extracting features from {file_path}: {e}")
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

# ---------------- MODERATE SINGLE FILE ----------------
def moderate_file(file_path, model, true_category=None):
    """Moderate a single audio file and return results"""
    try:
        # Extract features
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
        
        result = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "true_category": true_category,
            "predicted_category": predicted_category,
            "confidence": round(confidence, 2),
            "is_safe": is_safe,
            "safety_status": "SAFE" if is_safe else "UNSAFE",
            "all_probabilities": all_probabilities,
            "correct_prediction": true_category == predicted_category if true_category else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error moderating {file_path}: {e}")
        return None

# ---------------- BATCH MODERATION ----------------
def batch_moderate_test_data(model):
    """Moderate all test data and return results"""
    print("🔄 Starting batch moderation of test data...")
    
    all_results = []
    category_stats = {}
    
    # Initialize category stats
    for category in CATEGORIES:
        category_stats[category] = {
            "total_files": 0,
            "correct_predictions": 0,
            "safe_predictions": 0,
            "unsafe_predictions": 0
        }
    
    # Process each category
    for category in CATEGORIES:
        print(f"\n📁 Processing {category.upper()} category...")
        category_path = os.path.join(TEST_DATASET_PATH, category)
        
        if not os.path.exists(category_path):
            print(f"⚠️  Category path not found: {category_path}")
            continue
        
        files = [f for f in os.listdir(category_path) if f.lower().endswith(".mp3")]
        print(f"Found {len(files)} files in {category}")
        
        for file in files:
            file_path = os.path.join(category_path, file)
            result = moderate_file(file_path, model, true_category=category)
            
            if result:
                all_results.append(result)
                category_stats[category]["total_files"] += 1
                
                if result["correct_prediction"]:
                    category_stats[category]["correct_predictions"] += 1
                
                if result["is_safe"]:
                    category_stats[category]["safe_predictions"] += 1
                else:
                    category_stats[category]["unsafe_predictions"] += 1
    
    return all_results, category_stats

# ---------------- SAVE RESULTS ----------------
def save_batch_results(all_results, category_stats, output_file="batch_moderation_results.json"):
    """Save batch moderation results to JSON file"""
    try:
        # Calculate overall statistics
        total_files = len(all_results)
        correct_predictions = sum(1 for r in all_results if r["correct_prediction"])
        overall_accuracy = (correct_predictions / total_files * 100) if total_files > 0 else 0
        
        safe_files = sum(1 for r in all_results if r["true_category"] == "safe")
        unsafe_files = total_files - safe_files
        correct_safe = sum(1 for r in all_results if r["true_category"] == "safe" and r["is_safe"])
        correct_unsafe = sum(1 for r in all_results if r["true_category"] != "safe" and not r["is_safe"])
        
        # Create comprehensive results
        results_data = {
            "moderation_summary": {
                "total_files_processed": total_files,
                "overall_accuracy": round(overall_accuracy, 2),
                "safe_files_correctly_identified": correct_safe,
                "safe_files_total": safe_files,
                "safe_accuracy": round((correct_safe / safe_files * 100) if safe_files > 0 else 0, 2),
                "unsafe_files_correctly_identified": correct_unsafe,
                "unsafe_files_total": unsafe_files,
                "unsafe_accuracy": round((correct_unsafe / unsafe_files * 100) if unsafe_files > 0 else 0, 2),
                "timestamp": datetime.now().isoformat(),
                "model_used": MODEL_PATH
            },
            "category_statistics": category_stats,
            "detailed_results": all_results
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Batch results saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving batch results: {e}")
        return False

# ---------------- DISPLAY SUMMARY ----------------
def display_summary(all_results, category_stats):
    """Display a summary of batch moderation results"""
    total_files = len(all_results)
    correct_predictions = sum(1 for r in all_results if r["correct_prediction"])
    overall_accuracy = (correct_predictions / total_files * 100) if total_files > 0 else 0
    
    print("\n" + "="*70)
    print("📊 BATCH MODERATION SUMMARY")
    print("="*70)
    print(f"📁 Total files processed: {total_files}")
    print(f"🎯 Overall accuracy: {overall_accuracy:.2f}%")
    
    print(f"\n📈 Category-wise Results:")
    for category, stats in category_stats.items():
        accuracy = (stats["correct_predictions"] / stats["total_files"] * 100) if stats["total_files"] > 0 else 0
        print(f"  {category:>8}: {stats['correct_predictions']:>2}/{stats['total_files']:>2} ({accuracy:>5.1f}%)")
    
    # Safety analysis
    safe_files = sum(1 for r in all_results if r["true_category"] == "safe")
    unsafe_files = total_files - safe_files
    correct_safe = sum(1 for r in all_results if r["true_category"] == "safe" and r["is_safe"])
    correct_unsafe = sum(1 for r in all_results if r["true_category"] != "safe" and not r["is_safe"])
    
    print(f"\n🛡️  Safety Analysis:")
    print(f"  ✅ Safe files correctly identified: {correct_safe}/{safe_files} ({(correct_safe/safe_files*100) if safe_files > 0 else 0:.1f}%)")
    print(f"  ❌ Unsafe files correctly identified: {correct_unsafe}/{unsafe_files} ({(correct_unsafe/unsafe_files*100) if unsafe_files > 0 else 0:.1f}%)")
    print("="*70)

# ---------------- MAIN FUNCTION ----------------
def main():
    """Main batch moderation function"""
    print("🎵 Starting Batch Audio Moderation System")
    print("="*50)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Run batch moderation
    all_results, category_stats = batch_moderate_test_data(model)
    
    if not all_results:
        print("❌ No results generated")
        return
    
    # Display summary
    display_summary(all_results, category_stats)
    
    # Save results
    if save_batch_results(all_results, category_stats):
        print("✅ Batch moderation completed successfully!")
    else:
        print("⚠️  Batch moderation completed but failed to save results")

if __name__ == "__main__":
    main()

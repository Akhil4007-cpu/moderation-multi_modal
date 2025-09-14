#!/usr/bin/env python
"""
Create a working text classification model to replace the corrupted one
"""

import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

def create_text_model():
    """Create a simple text classification model for content moderation"""
    
    # Sample training data (in a real scenario, this would be much larger)
    training_texts = [
        # Safe content
        "This is a normal conversation about weather",
        "I love spending time with family",
        "Great movie recommendation, thanks!",
        "Looking forward to the weekend",
        "Nice photos from your vacation",
        "Happy birthday! Hope you have a great day",
        "Thanks for sharing this information",
        "The meeting is scheduled for tomorrow",
        "Please let me know if you need help",
        "Have a wonderful day",
        
        # Unsafe content (examples for training)
        "I want to hurt someone badly",
        "This makes me so angry I could kill",
        "Let's plan something violent",
        "I hate everyone and everything",
        "Time to get revenge on them",
        "Going to make them pay for this",
        "Violence is the only solution",
        "I'm going to destroy everything",
        "They deserve to suffer pain",
        "Planning to cause serious harm"
    ]
    
    # Labels: 0 = safe, 1 = unsafe
    labels = [0] * 10 + [1] * 10
    
    # Create a simple pipeline with TF-IDF and Naive Bayes
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('classifier', MultinomialNB())
    ])
    
    # Train the model
    model.fit(training_texts, labels)
    
    # Test the model
    test_texts = [
        "Hello, how are you today?",  # Should be safe
        "I want to kill someone"      # Should be unsafe
    ]
    
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)
    
    print("Model Training Complete!")
    print("Test Results:")
    for i, text in enumerate(test_texts):
        pred_label = "UNSAFE" if predictions[i] == 1 else "SAFE"
        confidence = max(probabilities[i]) * 100
        print(f"  Text: '{text}'")
        print(f"  Prediction: {pred_label} (confidence: {confidence:.1f}%)")
    
    return model

def save_model(model, filepath="text/text_model_fixed.pkl"):
    """Save the model using joblib for better compatibility"""
    try:
        joblib.dump(model, filepath)
        print(f"✅ Model saved successfully to {filepath}")
        return True
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False

def test_model_loading(filepath="text/text_model_fixed.pkl"):
    """Test loading the saved model"""
    try:
        loaded_model = joblib.load(filepath)
        
        # Test with sample text
        test_text = ["This is a test message"]
        prediction = loaded_model.predict(test_text)
        probability = loaded_model.predict_proba(test_text)
        
        print(f"✅ Model loaded successfully from {filepath}")
        print(f"Test prediction: {'UNSAFE' if prediction[0] == 1 else 'SAFE'}")
        print(f"Confidence: {max(probability[0]) * 100:.1f}%")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def main():
    """Main function to create and test the text model"""
    print("🔧 Creating Working Text Classification Model")
    print("=" * 50)
    
    # Create the model
    model = create_text_model()
    
    # Save the model
    if save_model(model):
        # Test loading
        test_model_loading()
        print("\n🎉 Text model creation complete!")
        print("The unified processor can now use ML-based text classification.")
    else:
        print("\n❌ Failed to create working text model.")

if __name__ == "__main__":
    main()

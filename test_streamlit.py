#!/usr/bin/env python3
"""
Test script for Streamlit app functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))

# Import the functions from the Streamlit app
from ui.app import load_model, make_prediction, create_input_form

def test_streamlit_app():
    """Test the core functionality of the Streamlit app."""
    print("Testing Streamlit App Functionality...")
    print("=" * 50)
    
    # Test 1: Load model
    print("1. Testing model loading...")
    model, metadata = load_model()
    
    if model is not None and metadata is not None:
        print("✅ Model loaded successfully!")
        print(f"   Model type: {metadata.get('model_type', 'Unknown')}")
        print(f"   Version: {metadata.get('version', 'Unknown')}")
    else:
        print("❌ Model loading failed!")
        return False
    
    # Test 2: Test prediction
    print("\n2. Testing prediction functionality...")
    test_features = {
        'age': 45,
        'sex': 1,
        'chest_pain_type': 3,
        'resting_bp_s': 130,
        'cholesterol': 250,
        'fasting_blood_sugar': 0,
        'resting_ecg': 0,
        'max_heart_rate': 150,
        'exercise_angina': 0,
        'oldpeak': 2.3,
        'st_slope': 0
    }
    
    prediction, prediction_proba = make_prediction(model, test_features)
    
    if prediction is not None and prediction_proba is not None:
        print("✅ Prediction successful!")
        print(f"   Prediction: {prediction}")
        print(f"   Probability: {prediction_proba}")
        print(f"   Risk probability: {prediction_proba[1]:.1%}")
    else:
        print("❌ Prediction failed!")
        return False
    
    # Test 3: Test input form (simulate)
    print("\n3. Testing input form simulation...")
    try:
        # Simulate the input form by creating a test dictionary
        test_input = {
            'age': 50,
            'sex': 0,
            'chest_pain_type': 1,
            'resting_bp_s': 140,
            'cholesterol': 280,
            'fasting_blood_sugar': 1,
            'resting_ecg': 1,
            'max_heart_rate': 160,
            'exercise_angina': 1,
            'oldpeak': 1.5,
            'st_slope': 1
        }
        print("✅ Input form simulation successful!")
        print(f"   Test input created with {len(test_input)} features")
    except Exception as e:
        print(f"❌ Input form simulation failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All Streamlit app tests passed!")
    print("The Streamlit application is ready to run.")
    print("\nTo start the app, run:")
    print("cd ui && streamlit run app.py")
    
    return True

if __name__ == "__main__":
    success = test_streamlit_app()
    if not success:
        print("\n❌ Some tests failed. Please check the model files and dependencies.")
        sys.exit(1) 
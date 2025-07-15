#!/usr/bin/env python3
"""
Heart Disease Model Export & Deployment Script
==============================================

This script exports the trained heart disease prediction model for deployment.
It creates a complete pipeline with preprocessing and saves all necessary components.

Author: Fadel
Date: 2025
"""

import pandas as pd
import numpy as np
import joblib
import json
import datetime
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def create_model_pipeline():
    """
    Create a complete pipeline with preprocessing and model.
    
    Returns:
        Pipeline: Complete sklearn pipeline
    """
    # Define preprocessing steps - Only scaling, no chi2 selection
    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Define complete pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        ))
    ])
    
    return model_pipeline

def load_and_prepare_data():
    """
    Load the selected features dataset and prepare for training.
    
    Returns:
        tuple: (X, y) features and target
    """
    try:
        # Load the selected features dataset
        df = pd.read_csv('data/heart_disease_selected.csv')
        X = df.drop('target', axis=1)
        y = df['target']
        
        print(f"Data loaded successfully!")
        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    except FileNotFoundError:
        print("Error: heart_disease_selected.csv not found in data/ directory")
        print("Please run the feature selection step first.")
        return None, None

def train_and_evaluate_model(X, y, model_pipeline):
    """
    Train the model and evaluate its performance.
    
    Args:
        X: Features
        y: Target
        model_pipeline: Sklearn pipeline
        
    Returns:
        dict: Performance metrics
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Train the pipeline
    print("Training model pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("Model training completed!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    return metrics

def save_model_and_metadata(model_pipeline, X, y, metrics):
    """
    Save the model pipeline and create metadata.
    
    Args:
        model_pipeline: Trained sklearn pipeline
        X: Features used for training
        y: Target used for training
        metrics: Performance metrics
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the complete pipeline
    model_path = 'models/heart_disease_pipeline.pkl'
    joblib.dump(model_pipeline, model_path)
    print(f"Model pipeline saved to: {model_path}")
    
    # Create model metadata
    model_metadata = {
        "model_name": "Heart Disease Classifier",
        "model_type": "Random Forest Pipeline",
        "version": "1.0",
        "created_date": datetime.datetime.now().isoformat(),
        "dataset_info": {
            "source": "UCI Heart Disease Dataset",
            "features_count": len(X.columns),
            "samples_count": len(X),
            "target_distribution": y.value_counts().to_dict()
        },
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42
        },
        "performance_metrics": metrics,
        "feature_names": X.columns.tolist(),
        "preprocessing_steps": [
            "StandardScaler"
        ],
        "pipeline_steps": list(model_pipeline.named_steps.keys())
    }
    
    # Save metadata
    metadata_path = 'models/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=4)
    print(f"Model metadata saved to: {metadata_path}")
    
    return model_path, metadata_path

def create_requirements_file():
    """
    Create requirements.txt file for deployment.
    """
    requirements = [
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    
    print("Requirements file created: requirements.txt")

def test_model_loading():
    """
    Test loading and using the saved model.
    
    Returns:
        bool: True if test successful, False otherwise
    """
    try:
        # Load the pipeline
        loaded_pipeline = joblib.load('models/heart_disease_pipeline.pkl')
        
        # Load the original data to get correct feature order
        df = pd.read_csv('data/heart_disease_selected.csv')
        X = df.drop('target', axis=1)
        
        # Create sample data with correct feature order
        sample_data = pd.DataFrame({
            'age': [45],
            'sex': [1],
            'chest_pain_type': [3],
            'resting_bp_s': [130],
            'cholesterol': [250],
            'fasting_blood_sugar': [0],
            'resting_ecg': [0],
            'max_heart_rate': [150],
            'exercise_angina': [0],
            'oldpeak': [2.3],
            'st_slope': [0]
        })
        
        # Ensure correct column order
        sample_data = sample_data[X.columns]
        
        # Make prediction
        prediction = loaded_pipeline.predict(sample_data)
        prediction_proba = loaded_pipeline.predict_proba(sample_data)
        
        print("✅ Model loading test successful!")
        print(f"   Prediction: {prediction[0]}")
        print(f"   Prediction probability: {prediction_proba[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def load_heart_disease_model():
    """
    Load the trained heart disease prediction model.
    
    Returns:
        tuple: (model_pipeline, metadata)
    """
    try:
        # Load the model pipeline
        model_pipeline = joblib.load('models/heart_disease_pipeline.pkl')
        
        # Load metadata
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        return model_pipeline, metadata
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_heart_disease(features_dict):
    """
    Make heart disease prediction using the trained model.
    
    Args:
        features_dict (dict): Dictionary containing patient features
        
    Returns:
        dict: Prediction results
    """
    try:
        model_pipeline, metadata = load_heart_disease_model()
        
        if model_pipeline is None:
            return {"error": "Model not loaded successfully"}
        
        # Load the original data to get correct feature order
        df = pd.read_csv('data/heart_disease_selected.csv')
        X = df.drop('target', axis=1)
        
        # Convert to DataFrame with correct column order
        features_df = pd.DataFrame([features_dict])
        features_df = features_df[X.columns]
        
        # Make prediction
        prediction = model_pipeline.predict(features_df)[0]
        prediction_proba = model_pipeline.predict_proba(features_df)[0]
        
        # Determine risk level
        prob_positive = prediction_proba[1]
        if prob_positive > 0.7:
            risk_level = "High"
        elif prob_positive > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "prediction": int(prediction),
            "probability": float(prob_positive),
            "risk_level": risk_level,
            "confidence": "High" if abs(prob_positive - 0.5) > 0.3 else "Medium"
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def main():
    """
    Main function to execute the model export and deployment process.
    """
    print("=" * 60)
    print("HEART DISEASE MODEL EXPORT & DEPLOYMENT")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    X, y = load_and_prepare_data()
    
    if X is None or y is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Step 2: Create model pipeline
    print("\n2. Creating model pipeline...")
    model_pipeline = create_model_pipeline()
    
    # Step 3: Train and evaluate model
    print("\n3. Training and evaluating model...")
    metrics = train_and_evaluate_model(X, y, model_pipeline)
    
    # Step 4: Save model and metadata
    print("\n4. Saving model and metadata...")
    model_path, metadata_path = save_model_and_metadata(model_pipeline, X, y, metrics)
    
    # Step 5: Create requirements file
    print("\n5. Creating requirements file...")
    create_requirements_file()
    
    # Step 6: Test model loading
    print("\n6. Testing model loading...")
    test_success = test_model_loading()
    
    # Step 7: Test prediction function
    print("\n7. Testing prediction function...")
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
    
    result = predict_heart_disease(test_features)
    print(f"   Prediction result: {result}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)
    print(f"✅ Model pipeline: {model_path}")
    print(f"✅ Model metadata: {metadata_path}")
    print(f"✅ Requirements file: requirements.txt")
    print(f"✅ Model loading test: {'PASSED' if test_success else 'FAILED'}")
    print(f"✅ Prediction function: {'WORKING' if 'error' not in result else 'FAILED'}")
    
    print("\n🎉 Model export and deployment completed successfully!")
    print("\nTo use the model in production:")
    print("1. Copy the 'models/' directory to your deployment environment")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Use the predict_heart_disease() function for predictions")

if __name__ == "__main__":
    main() 
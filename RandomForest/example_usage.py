#!/usr/bin/env python3

"""
Example script demonstrating Random Forest Binary Classification
for DDoS detection using Flatten LUCID data format
"""

import sys
import os
sys.path.append('.')
sys.path.append('../flatten_lucid')

from random_forest_binary import RandomForestBinaryClassifier
from rf_utils import *
import numpy as np

def example_training():
    """
    Example of training a Random Forest model
    """
    print("=== Random Forest Training Example ===")
    
    # Initialize classifier
    rf_classifier = RandomForestBinaryClassifier(n_estimators=100, max_depth=20)
    
    # Example using merged flatten dataset (adjust path as needed)
    train_path = "../DATASETS/merged_flatten_train_dataset.hdf5"
    
    if os.path.exists(train_path):
        print(f"Training with dataset: {train_path}")
        
        # Load training data
        X_train, y_train = rf_classifier.load_dataset(train_path)
        
        # Analyze dataset statistics
        stats = analyze_dataset_statistics(train_path)
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train the model
        rf_classifier.train(X_train_split, y_train_split, X_val, y_val)
        
        # Perform cross-validation
        cv_scores = rf_classifier.cross_validate(X_train_split, y_train_split, cv=5)
        
        # Get feature importance
        importance_df = rf_classifier.get_feature_importance()
        print("\\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        # Save model
        rf_classifier.save_model("example_rf_model.pkl")
        
        # Create performance report
        report, metrics = create_model_report(rf_classifier.model, X_val, y_val)
        print(report)
        
    else:
        print(f"Training dataset not found at: {train_path}")
        print("Please make sure you have run the merge_hdf5.py script to create merged datasets")

def example_hyperparameter_tuning():
    """
    Example of hyperparameter tuning
    """
    print("\\n=== Hyperparameter Tuning Example ===")
    
    # Initialize classifier
    rf_classifier = RandomForestBinaryClassifier()
    
    train_path = "../DATASETS/merged_flatten_train_dataset.hdf5"
    
    if os.path.exists(train_path):
        # Load training data
        X_train, y_train = rf_classifier.load_dataset(train_path)
        
        # Use a subset for faster tuning (optional)
        if len(X_train) > 50000:
            from sklearn.model_selection import train_test_split
            X_subset, _, y_subset, _ = train_test_split(
                X_train, y_train, train_size=50000, random_state=42, stratify=y_train
            )
            X_train, y_train = X_subset, y_subset
        
        # Perform hyperparameter tuning
        best_params = rf_classifier.tune_hyperparameters(
            X_train, y_train, method='random', n_iter=10
        )
        
        # Save tuned model
        rf_classifier.save_model("tuned_rf_model.pkl")
        
    else:
        print(f"Training dataset not found at: {train_path}")

def example_prediction():
    """
    Example of making predictions with a trained model
    """
    print("\\n=== Prediction Example ===")
    
    model_path = "example_rf_model.pkl"
    test_path = "../DATASETS/merged_flatten_test_dataset.hdf5"  # Adjust as needed
    
    if os.path.exists(model_path):
        # Load model
        rf_classifier = RandomForestBinaryClassifier()
        rf_classifier.load_model(model_path)
        
        # Create synthetic test data if real test data not available
        if not os.path.exists(test_path):
            print("Creating synthetic test data for demonstration...")
            # Generate some random test data with same dimensions as training
            np.random.seed(42)
            X_test = np.random.rand(1000, 21)  # 21 features for flatten format
            y_test = np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # 70% benign, 30% attack
        else:
            X_test, y_test = rf_classifier.load_dataset(test_path)
        
        # Make predictions
        predictions, probabilities = rf_classifier.predict(X_test)
        
        # Evaluate performance
        metrics = rf_classifier.evaluate(X_test, y_test)
        
        # Create detailed report
        report, _ = create_model_report(rf_classifier.model, X_test, y_test)
        print(report)
        
    else:
        print(f"Model file not found: {model_path}")
        print("Please run the training example first")

def compare_with_different_parameters():
    """
    Compare Random Forest performance with different parameters
    """
    print("\\n=== Parameter Comparison Example ===")
    
    train_path = "../DATASETS/merged_flatten_train_dataset.hdf5"
    
    if not os.path.exists(train_path):
        print(f"Training dataset not found at: {train_path}")
        return
    
    # Load a subset of data for faster comparison
    rf_temp = RandomForestBinaryClassifier()
    X_full, y_full = rf_temp.load_dataset(train_path)
    
    if len(X_full) > 10000:
        from sklearn.model_selection import train_test_split
        X_subset, _, y_subset, _ = train_test_split(
            X_full, y_full, train_size=10000, random_state=42, stratify=y_full
        )
        X_train, y_train = X_subset, y_subset
    else:
        X_train, y_train = X_full, y_full
    
    # Split for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_test, y_train_split, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )
    
    # Different configurations to test
    configs = {
        'RF_50_trees': {'n_estimators': 50, 'max_depth': None},
        'RF_100_trees': {'n_estimators': 100, 'max_depth': None},
        'RF_200_trees': {'n_estimators': 200, 'max_depth': None},
        'RF_100_depth_10': {'n_estimators': 100, 'max_depth': 10},
        'RF_100_depth_20': {'n_estimators': 100, 'max_depth': 20},
    }
    
    results = {}
    
    for config_name, params in configs.items():
        print(f"\\nTesting configuration: {config_name}")
        
        # Initialize and train classifier
        rf_classifier = RandomForestBinaryClassifier(**params)
        rf_classifier.train(X_train_split, y_train_split)
        
        # Evaluate
        metrics = rf_classifier.evaluate(X_test, y_test)
        results[config_name] = metrics
    
    # Compare results
    print("\\n=== Performance Comparison ===")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1_score']].round(4))

def main():
    """
    Run all examples
    """
    print("Random Forest Binary Classification for DDoS Detection")
    print("Compatible with Flatten LUCID data format")
    print("="*60)
    
    # Check if required packages are available
    try:
        import pandas as pd
        import sklearn
        import numpy as np
        print(f"✓ All required packages are available")
        print(f"  - NumPy: {np.__version__}")
        print(f"  - Scikit-learn: {sklearn.__version__}")
        print(f"  - Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        return
    
    # Run examples
    try:
        example_training()
        # example_hyperparameter_tuning()  # Uncomment for hyperparameter tuning
        example_prediction()
        # compare_with_different_parameters()  # Uncomment for parameter comparison
        
        print("\\n" + "="*60)
        print("Examples completed successfully!")
        print("Check the generated model files and output for results.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

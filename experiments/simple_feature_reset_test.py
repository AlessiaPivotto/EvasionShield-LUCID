#!/usr/bin/env python3
"""
Simple Feature Reset Test for TrafficManipulator datasets
Direct test using known dataset paths
"""

import os
import sys
import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_h5_dataset(filepath, max_samples=2000):
    """Load data from HDF5 file"""
    print(f"Loading: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        print(f"Available keys: {list(f.keys())}")
        
        # Get features and labels
        # Try different possible key names
        X = None
        y = None
        
        # Common key combinations
        key_combinations = [
            ('X', 'y'),
            ('set_x', 'set_y'),
            ('x', 'y'),
            ('data', 'labels'),
            ('features', 'targets')
        ]
        
        for x_key, y_key in key_combinations:
            if x_key in f and y_key in f:
                X = f[x_key][:]
                y = f[y_key][:]
                print(f"Using keys: {x_key}, {y_key}")
                break
        
        if X is None or y is None:
            # Try to use any available datasets
            keys = list(f.keys())
            if len(keys) >= 2:
                X = f[keys[0]][:]
                y = f[keys[1]][:]
                print(f"Using first two available keys: {keys[0]}, {keys[1]}")
            else:
                raise ValueError(f"Could not find suitable data in HDF5 file. Available keys: {keys}")
        
        print(f"Original data shape: X={X.shape}, y={y.shape}")
        print(f"Label distribution: {np.bincount(y)}")
        
        # Limit samples if requested
        if max_samples and X.shape[0] > max_samples:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X = X[indices]
            y = y[indices]
            print(f"Limited to {max_samples} samples")
        
        return X, y

def reset_features(X, feature_indices):
    """Reset specific features to zero"""
    X_reset = X.copy()
    X_reset[:, feature_indices] = 0
    return X_reset

def test_feature_reset():
    """Test feature reset on TrafficManipulator datasets"""
    
    # Define dataset paths
    datasets = {
        'baseline': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FLATTEN-PCAPS/10t-100n-DOS2019-flatten-dataset-train.hdf5',
        'manipulated': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42/09-TFTP/10t-100n-DOS2019-flatten-dataset-train.hdf5',
        'fragmented': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FRAGMENTED_42_150/09-TFTP/10t-100n-DOS2019-flatten-dataset-test.hdf5'
    }
    
    print("=== Feature Reset Evaluation ===")
    print(f"Testing {len(datasets)} datasets")
    
    all_results = []
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*50}")
        print(f"Testing {dataset_name.upper()} dataset")
        print(f"{'='*50}")
        
        try:
            if not os.path.exists(dataset_path):
                print(f"Dataset not found: {dataset_path}")
                continue
                
            # Load data
            X, y = load_h5_dataset(dataset_path)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train a simple classifier
            print("Training Random Forest classifier...")
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            
            # Baseline performance
            baseline_pred = clf.predict(X_test)
            baseline_accuracy = accuracy_score(y_test, baseline_pred)
            print(f"Baseline accuracy: {baseline_accuracy:.4f}")
            
            # Test different levels of feature reset
            reset_percentages = [10, 25, 50, 75]
            
            for reset_pct in reset_percentages:
                print(f"\nTesting {reset_pct}% random feature reset...")
                
                # Calculate number of features to reset
                num_features_to_reset = int(X.shape[1] * reset_pct / 100)
                feature_indices = np.random.choice(X.shape[1], num_features_to_reset, replace=False)
                
                # Reset features
                X_test_reset = reset_features(X_test, feature_indices)
                
                # Predict on reset data
                reset_pred = clf.predict(X_test_reset)
                reset_accuracy = accuracy_score(y_test, reset_pred)
                
                accuracy_drop = baseline_accuracy - reset_accuracy
                relative_drop = (accuracy_drop / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
                
                result = {
                    'dataset': dataset_name,
                    'reset_percentage': reset_pct,
                    'baseline_accuracy': baseline_accuracy,
                    'reset_accuracy': reset_accuracy,
                    'accuracy_drop': accuracy_drop,
                    'relative_drop': relative_drop,
                    'features_reset': num_features_to_reset
                }
                
                all_results.append(result)
                
                print(f"  Features reset: {num_features_to_reset}/{X.shape[1]}")
                print(f"  Reset accuracy: {reset_accuracy:.4f}")
                print(f"  Accuracy drop: {accuracy_drop:.4f} ({relative_drop:.1f}%)")
            
            # Test feature importance-based reset
            print(f"\nTesting feature importance-based reset...")
            feature_importance = clf.feature_importances_
            
            # Reset least important features (bottom 50%)
            num_features_to_reset = X.shape[1] // 2
            least_important = np.argsort(feature_importance)[:num_features_to_reset]
            
            X_test_least = reset_features(X_test, least_important)
            least_pred = clf.predict(X_test_least)
            least_accuracy = accuracy_score(y_test, least_pred)
            
            # Reset most important features (top 50%)
            most_important = np.argsort(feature_importance)[-num_features_to_reset:]
            
            X_test_most = reset_features(X_test, most_important)
            most_pred = clf.predict(X_test_most)
            most_accuracy = accuracy_score(y_test, most_pred)
            
            print(f"  Resetting LEAST important 50% features: {least_accuracy:.4f} (drop: {baseline_accuracy-least_accuracy:.4f})")
            print(f"  Resetting MOST important 50% features: {most_accuracy:.4f} (drop: {baseline_accuracy-most_accuracy:.4f})")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create results summary
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY OF RESULTS")
        print(f"{'='*60}")
        
        df = pd.DataFrame(all_results)
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Accuracy vs Reset Percentage
        plt.subplot(1, 3, 1)
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            plt.plot(data['reset_percentage'], data['reset_accuracy'], 
                    marker='o', label=dataset, linewidth=2, markersize=6)
        
        plt.xlabel('Features Reset (%)')
        plt.ylabel('Accuracy')
        plt.title('Model Performance vs Feature Reset')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy Drop
        plt.subplot(1, 3, 2)
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            plt.plot(data['reset_percentage'], data['accuracy_drop'], 
                    marker='s', label=dataset, linewidth=2, markersize=6)
        
        plt.xlabel('Features Reset (%)')
        plt.ylabel('Accuracy Drop')
        plt.title('Performance Degradation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Relative Drop
        plt.subplot(1, 3, 3)
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            plt.plot(data['reset_percentage'], data['relative_drop'], 
                    marker='^', label=dataset, linewidth=2, markersize=6)
        
        plt.xlabel('Features Reset (%)')
        plt.ylabel('Relative Performance Drop (%)')
        plt.title('Relative Performance Impact')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = '/home/rising/EvasionShield-LUCID/experiments/feature_reset_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nResults plot saved to: {output_path}")
        
        # Save detailed results
        csv_path = '/home/rising/EvasionShield-LUCID/experiments/feature_reset_detailed_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")
        
        plt.show()
    
    else:
        print("No results generated.")

if __name__ == "__main__":
    print("Starting Feature Reset Evaluation for TrafficManipulator datasets...")
    test_feature_reset()

#!/usr/bin/env python3
"""
Test script for Feature Reset Evaluation using TrafficManipulator datasets
Uses FLATTEN-PCAPS as baseline, MANIPULATED-FLATTEN_42 and FRAGMENTED_42_150

This script demonstrates how feature reset affects model performance across
different types of traffic manipulation.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import h5py
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directories to path to import LUCID modules
sys.path.append('/home/rising/EvasionShield-LUCID')
sys.path.append('/home/rising/EvasionShield-LUCID/flatten_lucid')
sys.path.append('/home/rising/EvasionShield-LUCID/experiments')

from feature_reset_evaluation import FeatureResetEvaluator

class TrafficManipulatorTester:
    def __init__(self):
        self.datasets = {
            'baseline': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FLATTEN-PCAPS',
            'manipulated': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42',
            'fragmented': '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FRAGMENTED_42_150'
        }
        self.results = {}
        
    def load_dataset(self, dataset_path, max_samples=1000):
        """Load dataset from HDF5 or CSV files in the given path"""
        print(f"\nLoading dataset from: {dataset_path}")
        
        # Look for HDF5 files first (in root directory)
        h5_files = glob.glob(os.path.join(dataset_path, "*.h5"))
        if h5_files:
            print(f"Found HDF5 files: {h5_files}")
            return self._load_h5_dataset(h5_files[0], max_samples)
        
        # Look for CSV files (in root directory)
        csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
        if csv_files:
            print(f"Found CSV files: {csv_files}")
            return self._load_csv_dataset(csv_files[0], max_samples)
        
        # Look for HDF5 files recursively in subdirectories
        h5_files_recursive = glob.glob(os.path.join(dataset_path, "**", "*.h5"), recursive=True)
        if h5_files_recursive:
            print(f"Found HDF5 files in subdirectories: {h5_files_recursive[:3]}...")  # Show first 3
            # Use train dataset if available, otherwise first file
            train_files = [f for f in h5_files_recursive if 'train' in f.lower()]
            if train_files:
                print(f"Using training dataset: {train_files[0]}")
                return self._load_h5_dataset(train_files[0], max_samples)
            else:
                return self._load_h5_dataset(h5_files_recursive[0], max_samples)
        
        # Look for CSV files recursively in subdirectories
        csv_files_recursive = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
        if csv_files_recursive:
            print(f"Found CSV files in subdirectories: {csv_files_recursive[:3]}...")
            return self._load_csv_dataset(csv_files_recursive[0], max_samples)
        
        raise FileNotFoundError(f"No HDF5 or CSV files found in {dataset_path}")
    
    def _load_h5_dataset(self, filepath, max_samples):
        """Load data from HDF5 file"""
        try:
            with h5py.File(filepath, 'r') as f:
                print(f"HDF5 file keys: {list(f.keys())}")
                
                # Try common key names
                for x_key in ['X', 'x', 'data', 'features', 'X_train', 'x_train']:
                    if x_key in f:
                        X = f[x_key][:]
                        break
                else:
                    # Use first available dataset
                    X = f[list(f.keys())[0]][:]
                
                # Try to find labels
                y = None
                for y_key in ['y', 'Y', 'labels', 'targets', 'y_train', 'Y_train']:
                    if y_key in f:
                        y = f[y_key][:]
                        break
                
                if y is None:
                    # Create dummy labels (assume binary classification)
                    y = np.random.randint(0, 2, size=X.shape[0])
                    print("No labels found, created dummy binary labels")
                
                # Limit samples if requested
                if max_samples and X.shape[0] > max_samples:
                    indices = np.random.choice(X.shape[0], max_samples, replace=False)
                    X = X[indices]
                    y = y[indices]
                
                print(f"Loaded data shape: X={X.shape}, y={y.shape if hasattr(y, 'shape') else len(y)}")
                return X, y
                
        except Exception as e:
            print(f"Error loading HDF5 file {filepath}: {e}")
            raise
    
    def _load_csv_dataset(self, filepath, max_samples):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            
            # Assume last column is label, rest are features
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            # Convert string labels to numeric if needed
            if y.dtype == 'object':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Limit samples if requested
            if max_samples and X.shape[0] > max_samples:
                indices = np.random.choice(X.shape[0], max_samples, replace=False)
                X = X[indices]
                y = y[indices]
            
            print(f"Loaded CSV data shape: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            print(f"Error loading CSV file {filepath}: {e}")
            raise
    
    def run_feature_reset_test(self, dataset_name, X, y):
        """Run feature reset evaluation on a dataset"""
        print(f"\n=== Running Feature Reset Test on {dataset_name.upper()} ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize evaluator
        evaluator = FeatureResetEvaluator()
        
        # Train a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Get baseline performance
        baseline_pred = model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        # Run feature reset evaluation
        print("Running feature reset evaluation...")
        
        # Test resetting different percentages of features
        reset_percentages = [10, 25, 50, 75]
        results = []
        
        for reset_pct in reset_percentages:
            print(f"\nTesting {reset_pct}% feature reset...")
            
            # Reset features
            X_reset = evaluator.reset_random_features(X_test.copy(), reset_percentage=reset_pct/100)
            
            # Evaluate
            reset_pred = model.predict(X_reset)
            reset_accuracy = accuracy_score(y_test, reset_pred)
            
            accuracy_drop = baseline_accuracy - reset_accuracy
            
            results.append({
                'dataset': dataset_name,
                'reset_percentage': reset_pct,
                'baseline_accuracy': baseline_accuracy,
                'reset_accuracy': reset_accuracy,
                'accuracy_drop': accuracy_drop,
                'relative_drop': accuracy_drop / baseline_accuracy if baseline_accuracy > 0 else 0
            })
            
            print(f"  Reset accuracy: {reset_accuracy:.4f}")
            print(f"  Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop/baseline_accuracy*100:.1f}%)")
        
        self.results[dataset_name] = results
        return results
    
    def test_specific_lucid_features(self, X, y, dataset_name):
        """Test resetting specific LUCID-relevant features"""
        print(f"\n=== Testing LUCID-specific features on {dataset_name} ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        baseline_pred = model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        
        # Define feature groups based on LUCID's network flow features
        feature_groups = {
            'temporal_features': list(range(0, min(10, X.shape[1]))),  # First 10 features (timing)
            'packet_size_features': list(range(10, min(20, X.shape[1]))),  # Next 10 features
            'flow_direction_features': list(range(20, min(30, X.shape[1]))),  # Next 10 features
            'statistical_features': list(range(30, min(40, X.shape[1]))),  # Next 10 features
        }
        
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        for group_name, feature_indices in feature_groups.items():
            if not feature_indices or max(feature_indices) >= X.shape[1]:
                continue
                
            print(f"\nTesting reset of {group_name}...")
            
            # Reset specific feature group
            X_reset = X_test.copy()
            X_reset[:, feature_indices] = 0
            
            # Evaluate
            reset_pred = model.predict(X_reset)
            reset_accuracy = accuracy_score(y_test, reset_pred)
            accuracy_drop = baseline_accuracy - reset_accuracy
            
            print(f"  Features reset: {len(feature_indices)} features (indices {min(feature_indices)}-{max(feature_indices)})")
            print(f"  Reset accuracy: {reset_accuracy:.4f}")
            print(f"  Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop/baseline_accuracy*100:.1f}%)")
    
    def run_all_tests(self):
        """Run tests on all three datasets"""
        print("=== TrafficManipulator Feature Reset Evaluation ===")
        print("Testing feature reset on three datasets:")
        print("1. FLATTEN-PCAPS (baseline)")
        print("2. MANIPULATED-FLATTEN_42 (manipulated traffic)")
        print("3. FRAGMENTED_42_150 (fragmented traffic)")
        
        all_results = []
        
        for dataset_name, dataset_path in self.datasets.items():
            try:
                print(f"\n{'='*60}")
                print(f"Processing {dataset_name.upper()} dataset")
                print(f"{'='*60}")
                
                # Load dataset
                X, y = self.load_dataset(dataset_path, max_samples=2000)
                
                # Run feature reset tests
                results = self.run_feature_reset_test(dataset_name, X, y)
                all_results.extend(results)
                
                # Test LUCID-specific features
                self.test_specific_lucid_features(X, y, dataset_name)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                continue
        
        # Create comparison plot
        if all_results:
            self.plot_comparison_results(all_results)
        
        return all_results
    
    def plot_comparison_results(self, results):
        """Plot comparison of feature reset impact across datasets"""
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Accuracy drop vs reset percentage
        plt.subplot(2, 2, 1)
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            plt.plot(data['reset_percentage'], data['accuracy_drop'], 
                    marker='o', label=dataset, linewidth=2, markersize=8)
        
        plt.xlabel('Features Reset (%)')
        plt.ylabel('Accuracy Drop')
        plt.title('Accuracy Drop vs Feature Reset Percentage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Relative accuracy drop
        plt.subplot(2, 2, 2)
        for dataset in df['dataset'].unique():
            data = df[df['dataset'] == dataset]
            plt.plot(data['reset_percentage'], data['relative_drop'] * 100, 
                    marker='s', label=dataset, linewidth=2, markersize=8)
        
        plt.xlabel('Features Reset (%)')
        plt.ylabel('Relative Accuracy Drop (%)')
        plt.title('Relative Performance Degradation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Heatmap of accuracy drops
        plt.subplot(2, 2, 3)
        pivot_df = df.pivot(index='dataset', columns='reset_percentage', values='accuracy_drop')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='Reds', cbar=True)
        plt.title('Accuracy Drop Heatmap')
        plt.ylabel('Dataset Type')
        plt.xlabel('Features Reset (%)')
        
        # Plot 4: Bar chart comparison at 50% reset
        plt.subplot(2, 2, 4)
        reset_50_data = df[df['reset_percentage'] == 50]
        if not reset_50_data.empty:
            plt.bar(reset_50_data['dataset'], reset_50_data['accuracy_drop'])
            plt.ylabel('Accuracy Drop')
            plt.title('Accuracy Drop at 50% Feature Reset')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('/home/rising/EvasionShield-LUCID/experiments/feature_reset_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nComparison plot saved to: /home/rising/EvasionShield-LUCID/experiments/feature_reset_comparison.png")

def main():
    """Main function to run the tests"""
    tester = TrafficManipulatorTester()
    
    print("Starting TrafficManipulator Feature Reset Evaluation...")
    print("This will test how feature reset affects model performance")
    print("on baseline, manipulated, and fragmented traffic datasets.\n")
    
    try:
        results = tester.run_all_tests()
        
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        
        if results:
            df = pd.DataFrame(results)
            print(df.to_string(index=False))
            
            print(f"\nDetailed results saved to: /home/rising/EvasionShield-LUCID/experiments/")
            print("Check the generated plot for visual comparison.")
        else:
            print("No results generated. Check dataset paths and formats.")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

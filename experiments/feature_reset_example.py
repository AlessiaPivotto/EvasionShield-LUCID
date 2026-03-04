#!/usr/bin/env python3
"""
Example: Feature Reset Evaluation for LUCID
===========================================

This example demonstrates how to perform feature reset (zeroing) evaluation
on your LUCID models to understand feature importance and model robustness.

Usage Examples:
--------------

1. Evaluate a CNN model:
   python feature_reset_example.py --model /path/to/model.h5 --data /path/to/test.hdf5

2. Evaluate a flatten model:
   python feature_reset_example.py --model ./MLP/mlp_flatten_binary_classification.h5 --data ./DATASETS/merged_flatten_train_dataset.hdf5

3. Only run individual feature analysis:
   python feature_reset_example.py --model model.h5 --data test.hdf5 --individual_only
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add the parent directories to the path so we can import LUCID modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir / "lucid-ddos-master"))
sys.path.append(str(parent_dir / "flatten_lucid"))
sys.path.append(str(parent_dir / "multiclass-lucid"))

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} found")
    except ImportError:
        print("✗ TensorFlow not found. Please install: pip install tensorflow")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib found")
    except ImportError:
        print("✗ Matplotlib not found. Please install: pip install matplotlib")
        return False
    
    try:
        import seaborn as sns
        print("✓ Seaborn found")
    except ImportError:
        print("✗ Seaborn not found. Please install: pip install seaborn")
        return False
    
    try:
        import h5py
        print("✓ h5py found")
    except ImportError:
        print("✗ h5py not found. Please install: pip install h5py")
        return False
    
    return True

def find_available_models_and_data():
    """Find available models and test data in the workspace."""
    print("\n🔍 Scanning workspace for available models and datasets...")
    
    models = []
    datasets = []
    
    # Look for models
    for model_path in Path(".").rglob("*.h5"):
        if model_path.stat().st_size > 1024:  # At least 1KB
            models.append(model_path)
    
    # Look for datasets
    for data_path in Path(".").rglob("*.hdf5"):
        if "test" in data_path.name.lower() and data_path.stat().st_size > 1024:
            datasets.append(data_path)
    
    print(f"\n📊 Found {len(models)} model files:")
    for i, model in enumerate(models[:10]):  # Show first 10
        print(f"  {i+1}. {model}")
    if len(models) > 10:
        print(f"  ... and {len(models)-10} more")
    
    print(f"\n📁 Found {len(datasets)} test datasets:")
    for i, dataset in enumerate(datasets[:10]):  # Show first 10
        print(f"  {i+1}. {dataset}")
    if len(datasets) > 10:
        print(f"  ... and {len(datasets)-10} more")
    
    return models, datasets

def simple_feature_reset_demo(model_path, data_path):
    """
    Simplified demonstration of feature reset evaluation.
    """
    print(f"\n🚀 Running Feature Reset Demo")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        import h5py
        from sklearn.metrics import accuracy_score, f1_score
        
        # Load model
        print("\n1. Loading model...")
        model = load_model(model_path)
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        
        # Load test data
        print("\n2. Loading test data...")
        with h5py.File(data_path, 'r') as f:
            print(f"   Available keys in HDF5: {list(f.keys())}")
            
            # Try different common key names
            x_key = 'x' if 'x' in f.keys() else 'X'
            y_key = 'y' if 'y' in f.keys() else 'Y'
            
            if x_key in f.keys() and y_key in f.keys():
                X_test = f[x_key][:]
                Y_test = f[y_key][:]
                print(f"   Test data shape: {X_test.shape}")
                print(f"   Test labels shape: {Y_test.shape}")
            else:
                print(f"   ⚠️  Could not find standard data keys. Available: {list(f.keys())}")
                return
        
        # Ensure data compatibility with model
        if len(X_test.shape) == 3 and len(model.input_shape) == 4:
            print("   🔧 Expanding dimensions for CNN model...")
            X_test = np.expand_dims(X_test, axis=-1)
        
        # Limit to first 1000 samples for quick demo
        if len(X_test) > 1000:
            print("   📊 Using first 1000 samples for quick demo...")
            X_test = X_test[:1000]
            Y_test = Y_test[:1000]
        
        # Get baseline performance
        print("\n3. Evaluating baseline performance...")
        baseline_pred = model.predict(X_test, batch_size=256, verbose=0)
        
        if len(baseline_pred.shape) > 1 and baseline_pred.shape[1] > 1:
            # Multiclass
            y_pred_baseline = np.argmax(baseline_pred, axis=1)
            y_true = Y_test if len(Y_test.shape) == 1 else np.argmax(Y_test, axis=1)
        else:
            # Binary
            y_pred_baseline = (baseline_pred > 0.5).astype(int).flatten()
            y_true = Y_test.flatten()
        
        baseline_accuracy = accuracy_score(y_true, y_pred_baseline)
        baseline_f1 = f1_score(y_true, y_pred_baseline, average='weighted')
        
        print(f"   ✅ Baseline Accuracy: {baseline_accuracy:.4f}")
        print(f"   ✅ Baseline F1-Score: {baseline_f1:.4f}")
        
        # Feature reset analysis
        print("\n4. Running feature reset analysis...")
        
        results = []
        
        if len(X_test.shape) == 4:  # CNN model
            print("   🔍 Analyzing CNN features...")
            height, width = X_test.shape[1:3]
            
            # Sample only a few features for demo
            sample_features = min(20, height * width)
            feature_indices = np.random.choice(height * width, sample_features, replace=False)
            
            for idx, feat_idx in enumerate(feature_indices):
                i, j = feat_idx // width, feat_idx % width
                print(f"   Processing feature ({i},{j}) - {idx+1}/{sample_features}")
                
                # Create modified data with feature zeroed
                X_modified = X_test.copy()
                X_modified[:, i, j] = 0
                
                # Evaluate
                pred_modified = model.predict(X_modified, batch_size=256, verbose=0)
                
                if len(pred_modified.shape) > 1 and pred_modified.shape[1] > 1:
                    y_pred_modified = np.argmax(pred_modified, axis=1)
                else:
                    y_pred_modified = (pred_modified > 0.5).astype(int).flatten()
                
                acc_modified = accuracy_score(y_true, y_pred_modified)
                f1_modified = f1_score(y_true, y_pred_modified, average='weighted')
                
                accuracy_drop = baseline_accuracy - acc_modified
                f1_drop = baseline_f1 - f1_modified
                
                results.append({
                    'feature': f'({i},{j})',
                    'accuracy': acc_modified,
                    'f1_score': f1_modified,
                    'accuracy_drop': accuracy_drop,
                    'f1_drop': f1_drop,
                    'importance': (accuracy_drop + f1_drop) / 2
                })
        
        elif len(X_test.shape) == 2:  # Flatten model
            print("   🔍 Analyzing flatten features...")
            num_features = X_test.shape[1]
            
            # Sample only a few features for demo
            sample_features = min(20, num_features)
            feature_indices = np.random.choice(num_features, sample_features, replace=False)
            
            for idx, feat_idx in enumerate(feature_indices):
                print(f"   Processing feature {feat_idx} - {idx+1}/{sample_features}")
                
                # Create modified data with feature zeroed
                X_modified = X_test.copy()
                X_modified[:, feat_idx] = 0
                
                # Evaluate
                pred_modified = model.predict(X_modified, batch_size=256, verbose=0)
                
                if len(pred_modified.shape) > 1 and pred_modified.shape[1] > 1:
                    y_pred_modified = np.argmax(pred_modified, axis=1)
                else:
                    y_pred_modified = (pred_modified > 0.5).astype(int).flatten()
                
                acc_modified = accuracy_score(y_true, y_pred_modified)
                f1_modified = f1_score(y_true, y_pred_modified, average='weighted')
                
                accuracy_drop = baseline_accuracy - acc_modified
                f1_drop = baseline_f1 - f1_modified
                
                results.append({
                    'feature': feat_idx,
                    'accuracy': acc_modified,
                    'f1_score': f1_modified,
                    'accuracy_drop': accuracy_drop,
                    'f1_drop': f1_drop,
                    'importance': (accuracy_drop + f1_drop) / 2
                })
        
        # Show results
        print(f"\n📊 Feature Reset Analysis Results:")
        print("=" * 60)
        
        if results:
            # Sort by importance
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('importance', ascending=False)
            
            print(f"{'Feature':<12} {'Accuracy':<10} {'F1-Score':<10} {'Acc Drop':<10} {'F1 Drop':<10} {'Importance':<12}")
            print("-" * 60)
            
            for _, row in results_df.head(10).iterrows():
                print(f"{str(row['feature']):<12} {row['accuracy']:<10.4f} {row['f1_score']:<10.4f} {row['accuracy_drop']:<10.4f} {row['f1_drop']:<10.4f} {row['importance']:<12.4f}")
            
            # Save results
            output_file = f"feature_reset_demo_results.csv"
            results_df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")
            
            # Summary statistics
            print(f"\n📈 Summary Statistics:")
            print(f"   Mean importance score: {results_df['importance'].mean():.4f}")
            print(f"   Max importance score: {results_df['importance'].max():.4f}")
            print(f"   Features with high impact (>mean): {len(results_df[results_df['importance'] > results_df['importance'].mean()])}")
            
            print(f"\n🎯 Key Insights:")
            most_important = results_df.iloc[0]
            print(f"   • Most important feature: {most_important['feature']} (importance: {most_important['importance']:.4f})")
            print(f"   • Zeroing this feature causes {most_important['accuracy_drop']:.1%} accuracy drop")
            print(f"   • Average feature importance: {results_df['importance'].mean():.4f}")
            
        print(f"\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function with user-friendly interface."""
    print("🧪 LUCID Feature Reset Evaluation Demo")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(
        description="Feature Reset Evaluation Demo for LUCID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('-m', '--model', help="Path to trained model (.h5)")
    parser.add_argument('-d', '--data', help="Path to test dataset (.hdf5)")
    parser.add_argument('--scan', action='store_true', help="Scan workspace for available files")
    parser.add_argument('--check-deps', action='store_true', help="Check dependencies")
    
    args = parser.parse_args()
    
    if args.check_deps:
        if check_dependencies():
            print("\n✅ All dependencies are available!")
        else:
            print("\n❌ Some dependencies are missing. Please install them.")
        return
    
    if args.scan or (not args.model and not args.data):
        models, datasets = find_available_models_and_data()
        
        if not args.model and models:
            print(f"\n💡 Suggestion: Try running with a model:")
            print(f"   python {sys.argv[0]} --model {models[0]} --data <dataset>")
        
        if not args.data and datasets:
            print(f"\n💡 Suggestion: Try running with a dataset:")
            print(f"   python {sys.argv[0]} --model <model> --data {datasets[0]}")
        
        if models and datasets:
            print(f"\n🎯 Complete example command:")
            print(f"   python {sys.argv[0]} --model {models[0]} --data {datasets[0]}")
        
        return
    
    if not args.model or not args.data:
        print("❌ Both --model and --data arguments are required")
        print("Use --scan to find available files")
        return
    
    # Check if files exist
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        return
    
    # Check dependencies before running
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before running the demo.")
        return
    
    # Run the demo
    simple_feature_reset_demo(args.model, args.data)

if __name__ == "__main__":
    main()

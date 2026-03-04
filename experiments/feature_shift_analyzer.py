#!/usr/bin/env python3
"""
Feature Distribution Comparison Analysis
Analyzes and visualizes how feature distributions change across:
1. Original (Baseline)
2. Manipulated (PSO)
3. Fragmented (PSO + Fragmentation)
"""

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import argparse

# Feature names mapping for better visualization
FEATURE_NAMES = {
    0: 'Flow Duration',
    1: 'Fwd Pkt Len Mean',
    2: 'Bwd Pkt Len Mean', 
    3: 'Tot Len Fwd Pkts',
    4: 'Tot Len Bwd Pkts',
    5: 'Tot Fwd Pkts',
    6: 'Tot Bwd Pkts', 
    7: 'Flow Byts/s',
    8: 'Flow Pkts/s',
    9: 'Flow IAT Std', 
    10: 'Flow IAT Max',
    11: 'Fwd IAT Std',
    12: 'Fwd IAT Max', 
    13: 'Bwd IAT Std',
    14: 'Bwd IAT Max',
    15: 'Flow IAT Mean', # The critical one identified
    16: 'Fwd IAT Mean',
    17: 'Bwd IAT Mean',
    18: 'Fwd Pkt Len Std',
    19: 'Bwd Pkt Len Std',
    20: 'Pkt Len Var'
}

CRITICAL_FEATURES = [15, 0, 2] # Features identified in previous analysis

def load_dataset(filepath, dataset_type):
    """Load data from HDF5 file"""
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return None
        
    try:
        with h5py.File(filepath, 'r') as f:
            # Check keys
            if 'set_x' in f:
                X = f['set_x'][:]
                y = f['set_y'][:] if 'set_y' in f else None
            elif 'X' in f:
                X = f['X'][:]
                y = f['y'][:] if 'y' in f else None
            else:
                print(f"Unknown structure in {filepath}. Keys: {list(f.keys())}")
                return None
            
            # Filter only malicious samples (assuming label 1 is malicious)
            # If y is provided, use it. Otherwise assume all are malicious for manipulated/fragmented
            if y is not None:
                # For baseline, we want to separate by class if needed, but usually we compare attacks
                # Let's assume we want to analyze the malicious traffic features
                malicious_indices = np.where(y == 1)[0]
                if len(malicious_indices) > 0:
                    X = X[malicious_indices]
            
            return X
            
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_feature_changes(features, feature_idx, labels):
    """
    Analyze statistical changes for a specific feature
    features: list of feature arrays [baseline, manipulated, fragmented]
    """
    stats = []
    
    for i, data in enumerate(features):
        if data is None or len(data) == 0:
            stats.append({'mean': 0, 'std': 0, 'median': 0})
            continue
            
        feat_data = data[:, feature_idx]
        stats.append({
            'scenario': labels[i],
            'mean': np.mean(feat_data),
            'std': np.std(feat_data),
            'median': np.median(feat_data),
            'min': np.min(feat_data),
            'max': np.max(feat_data)
        })
        
    return pd.DataFrame(stats)

def plot_feature_distributions(datasets, labels, feature_indices, output_dir):
    """
    Create distribution plots for selected features
    datasets: list of X arrays
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Prepare data for plotting
    for feat_idx in feature_indices:
        feat_name = FEATURE_NAMES.get(feat_idx, f"Feature {feat_idx}")
        
        plt.figure(figsize=(12, 6))
        
        # Create KDE plot for each scenario
        for i, data in enumerate(datasets):
            if data is None or len(data) == 0:
                continue
                
            # Extract feature column
            feat_data = data[:, feat_idx]
            
            # Handle potential outliers for better visualization (clip to 95th percentile)
            upper_limit = np.percentile(feat_data, 95)
            # Avoid clipping if data is very sparse or all zeros
            if upper_limit > 0:
                viz_data = feat_data[feat_data <= upper_limit]
            else:
                viz_data = feat_data
                
            sns.kdeplot(viz_data, label=f"{labels[i]} (Mean: {np.mean(feat_data):.2f})", fill=True, alpha=0.3)
            
        plt.title(f'Distribution Shift: {feat_name}')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = os.path.join(output_dir, f"feature_{feat_idx}_distribution.png")
        plt.savefig(filename)
        print(f"Saved distribution plot to {filename}")
        plt.close()

def plot_grouped_boxplot(datasets, labels, feature_indices, output_dir):
    """
    Create a grouped boxplot comparing specific features across scenarios
    """
    data_list = []
    
    for feat_idx in feature_indices:
        feat_name = FEATURE_NAMES.get(feat_idx, f"Feat {feat_idx}")
        
        for i, data in enumerate(datasets):
            if data is None:
                continue
                
            feat_vals = data[:, feat_idx]
            
            # Sample if too large
            if len(feat_vals) > 1000:
                feat_vals = np.random.choice(feat_vals, 1000, replace=False)
                
            # Normalize for comparison? Or raw values?
            # Using raw values first. Boxplots handle different scales cleanly if done per feature
            # But combining them is hard. Let's do individual feature boxplots
            
            for val in feat_vals:
                data_list.append({
                    'Feature': feat_name,
                    'Value': val,
                    'Scenario': labels[i]
                })
    
    df = pd.DataFrame(data_list)
    
    # Create separate boxplot for each feature since scales vary wildly
    for feat_idx in feature_indices:
        feat_name = FEATURE_NAMES.get(feat_idx, f"Feat {feat_idx}")
        dataset_subset = df[df['Feature'] == feat_name]
        
        if len(dataset_subset) == 0:
            continue
            
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Scenario', y='Value', data=dataset_subset)
        plt.title(f'Value Range Comparison: {feat_name}')
        plt.grid(True, axis='y', alpha=0.3)
        
        filename = os.path.join(output_dir, f"feature_{feat_idx}_boxplot.png")
        plt.savefig(filename)
        plt.close()

def main():
    # Use absolute paths to resolve "File not found" issues
    workspace_root = "/home/rising/EvasionShield-LUCID"
    base_dir_orig = os.path.join(workspace_root, "TrafficManipulator-master/DATASETS/FLATTEN-PCAPS")
    base_dir_manip = os.path.join(workspace_root, "TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42")
    base_dir_frag = os.path.join(workspace_root, "TrafficManipulator-master/DATASETS/FRAGMENTED_42_150")
    
    # Let's focus on a representative attack type, e.g. WebDDoS or Syn
    # We can iterate through a few or just pick one
    target_attack = "00-WebDDoS" # Consistent with your previous example
    
    print(f"Analyzing feature shifts for {target_attack}...")
    
    # Define paths
    path_orig = os.path.join(base_dir_orig, target_attack, "10t-100n-DOS2019-flatten-dataset-test.hdf5")
    path_manip = os.path.join(base_dir_manip, target_attack, "10t-100n-DOS2019-flatten-dataset-test.hdf5")
    path_frag = os.path.join(base_dir_frag, target_attack, "10t-100n-DOS2019-flatten-dataset-test.hdf5")
    
    # Load data
    X_orig = load_dataset(path_orig, "Original")
    X_manip = load_dataset(path_manip, "Manipulated")
    X_frag = load_dataset(path_frag, "Fragmented")
    
    datasets = [X_orig, X_manip, X_frag]
    labels = ["Baseline", "Manipulated", "Fragmented"]
    
    # Remove None datasets
    valid_indices = [i for i, d in enumerate(datasets) if d is not None]
    datasets = [datasets[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    if not datasets:
        print("No datasets loaded!")
        return

    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Generate statistics table
    stats_dfs = []
    
    print("\nStatistical Analysis of Key Features:")
    print("-" * 60)
    
    for feat_idx in CRITICAL_FEATURES:
        feat_name = FEATURE_NAMES.get(feat_idx, f"Feature {feat_idx}")
        df = analyze_feature_changes(datasets, feat_idx, labels)
        df['Feature'] = feat_name
        df['Feature ID'] = feat_idx
        stats_dfs.append(df)
        
        print(f"\nFeature: {feat_name} ({feat_idx})")
        print(df[['scenario', 'mean', 'std', 'median']].to_string(index=False))
        
    # Save full stats
    full_stats = pd.concat(stats_dfs)
    full_stats.to_csv(os.path.join(output_dir, 'feature_shift_statistics.csv'), index=False)
    
    # Generate Plots
    print("\nGenerating visualizations...")
    plot_feature_distributions(datasets, labels, CRITICAL_FEATURES, output_dir)
    plot_grouped_boxplot(datasets, labels, CRITICAL_FEATURES, output_dir)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()

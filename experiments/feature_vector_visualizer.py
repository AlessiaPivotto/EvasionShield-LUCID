#!/usr/bin/env python3
"""
Feature Vector Visualizer
Visualizes the actual feature vectors (values) comparing Original vs Manipulated vs Fragmented.
Uses Mean Vectors and representative samples.
"""

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import pi

# Feature names mapping
FEATURE_NAMES = [
    'Flow Duration', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Tot Len Fwd Pkts',
    'Tot Len Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Byts/s',
    'Flow Pkts/s', 'Flow IAT Std', 'Flow IAT Max', 'Fwd IAT Std',
    'Fwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Max', 'Flow IAT Mean',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Std',
    'Pkt Len Var'
]

def load_dataset(filepath):
    """Load data from HDF5 file"""
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return None
        
    try:
        with h5py.File(filepath, 'r') as f:
            if 'set_x' in f:
                X = f['set_x'][:]
            elif 'X' in f:
                X = f['X'][:]
            else:
                return None
            return X
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def normalize_features(means_list):
    """
    Normalize features for visualization.
    Each feature is normalized relative to the maximum value observed across all scenarios.
    This preserves relative changes while bringing all features to [0, 1] scale.
    """
    means_array = np.array(means_list) # shape (3, 21)
    
    # Find max for each feature across the 3 scenarios
    max_vals = np.max(means_array, axis=0)
    
    # Avoid division by zero
    max_vals[max_vals == 0] = 1.0
    
    normalized = means_array / max_vals
    return normalized, max_vals

def plot_radar_chart(normalized_means, labels, output_path):
    """Create a radar chart comparing the feature vectors"""
    # Number of variables
    N = len(FEATURE_NAMES)
    
    # What will be the angle of each axis in the plot?
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], FEATURE_NAMES, color='grey', size=8)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot each scenario
    colors = ['g', 'b', 'r']
    for i, (mean_vec, label) in enumerate(zip(normalized_means, labels)):
        values = mean_vec.tolist()
        values += values[:1] # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Feature Vector Comparison (Normalized relative to Max)", size=15, y=1.1)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved radar chart to {output_path}")
    plt.close()

def plot_heatmap_comparison(means_list, labels, output_path):
    """Create a heatmap comparison"""
    # Transpose so features are rows, scenarios are columns
    data = np.array(means_list).T
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=labels, index=FEATURE_NAMES)
    
    # Normalize row-wise for heatmap (so we can see relative drop per feature)
    # Each row (feature) is scaled 0-1
    df_norm = df.div(df.max(axis=1), axis=0)
    df_norm = df_norm.fillna(0)
    
    plt.figure(figsize=(8, 12))
    sns.heatmap(df_norm, annot=True, cmap="RdYlGn", fmt=".2f", 
                cbar_kws={'label': 'Relative Value (Normalized to Max)'})
    
    plt.title("Feature Value Degradation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved heatmap to {output_path}")
    plt.close()

def plot_bar_comparison(means_list, labels, output_path):
    """Create a grouped bar chart for detailed comparison"""
    normalized, _ = normalize_features(means_list)
    
    n_features = len(FEATURE_NAMES)
    x = np.arange(n_features)
    width = 0.25
    
    plt.figure(figsize=(18, 8))
    
    plt.bar(x - width, normalized[0], width, label=labels[0], color='green', alpha=0.7)
    plt.bar(x, normalized[1], width, label=labels[1], color='blue', alpha=0.7)
    plt.bar(x + width, normalized[2], width, label=labels[2], color='red', alpha=0.7)
    
    plt.ylabel('Normalized Feature Value')
    plt.title('Feature Value Comparison by Attack Phase')
    plt.xticks(x, FEATURE_NAMES, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved bar chart to {output_path}")
    plt.close()

def main():
    # Paths (using absolute paths as fixed previously)
    workspace_root = "/home/rising/EvasionShield-LUCID"
    base_dir_orig = os.path.join(workspace_root, "TrafficManipulator-master/DATASETS/FLATTEN-PCAPS")
    base_dir_manip = os.path.join(workspace_root, "TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42")
    base_dir_frag = os.path.join(workspace_root, "TrafficManipulator-master/DATASETS/FRAGMENTED_42_150")
    
    target_attack = "00-WebDDoS"
    
    path_orig = os.path.join(base_dir_orig, target_attack, "10t-100n-DOS2019-flatten-dataset-test.hdf5")
    path_manip = os.path.join(base_dir_manip, target_attack, "10t-100n-DOS2019-flatten-dataset-test.hdf5")
    path_frag = os.path.join(base_dir_frag, target_attack, "10t-100n-DOS2019-flatten-dataset-test.hdf5")
    
    print(f"Loading datasets for {target_attack}...")
    X_orig = load_dataset(path_orig)
    X_manip = load_dataset(path_manip)
    X_frag = load_dataset(path_frag)
    
    if X_orig is None or X_manip is None or X_frag is None:
        print("Failed to load one or more datasets.")
        return
        
    # Calculate Mean Vectors
    mean_orig = np.mean(X_orig, axis=0)
    mean_manip = np.mean(X_manip, axis=0)
    mean_frag = np.mean(X_frag, axis=0)
    
    means = [mean_orig, mean_manip, mean_frag]
    labels = ["Original", "Manipulated", "Fragmented"]
    
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    norm_means, _ = normalize_features(means)
    
    # Generate Visualizations
    plot_radar_chart(norm_means, labels, os.path.join(output_dir, "feature_vector_radar.png"))
    plot_heatmap_comparison(means, labels, os.path.join(output_dir, "feature_vector_heatmap.png"))
    plot_bar_comparison(means, labels, os.path.join(output_dir, "feature_vector_bars.png"))
    
    # Save the actual values to CSV for reference
    df_values = pd.DataFrame(np.array(means).T, columns=labels, index=FEATURE_NAMES)
    df_values.to_csv(os.path.join(output_dir, "feature_vector_values.csv"))
    print(f"Saved raw values to {output_dir}/feature_vector_values.csv")

if __name__ == "__main__":
    main()

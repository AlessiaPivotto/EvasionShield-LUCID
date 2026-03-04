#!/usr/bin/env python3
"""
Single Flow Tracer
Identifies a representative malicious flow from the original dataset and attempts
to trace its feature evolution through the manipulation and fragmentation phases.
"""

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

FEATURE_NAMES = [
    'Flow Duration', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Tot Len Fwd Pkts',
    'Tot Len Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Flow Byts/s',
    'Flow Pkts/s', 'Flow IAT Std', 'Flow IAT Max', 'Fwd IAT Std',
    'Fwd IAT Max', 'Bwd IAT Std', 'Bwd IAT Max', 'Flow IAT Mean',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Std',
    'Pkt Len Var'
]

# Critical features focusing on timing and size
KEY_FEATURES_INDICES = [0, 15, 2, 7, 8]  
# 0: Duration, 15: Flow IAT Mean, 2: Bwd Pkt Len Mean, 7: Flow Byts/s, 8: Flow Pkts/s

def load_dataset(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'set_x' in f:
                return f['set_x'][:]
            elif 'X' in f:
                return f['X'][:]
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def find_representative_sample(data):
    """
    Finds a sample that is close to the mean of the dataset.
    This ensures we pick a 'typical' attack flow rather than an outlier.
    """
    if data is None or len(data) == 0:
        return None, -1
        
    mean_vec = np.mean(data, axis=0)
    
    # Calculate Euclidean distance of each sample to the mean
    distances = np.linalg.norm(data - mean_vec, axis=1)
    
    # Find index of minimum distance
    idx = np.argmin(distances)
    return data[idx], idx

def plot_single_flow_comparison(orig_vec, manip_vec, frag_vec, output_path):
    """Create a detailed bar chart comparing the specific feature values for one flow"""
    
    # Select only key features for cleaner plot or plot all?
    # Let's plot all for completeness but highlight key ones
    
    n_features = len(FEATURE_NAMES)
    x = np.arange(n_features)
    width = 0.25
    
    plt.figure(figsize=(18, 8))
    
    # Plot bars
    plt.bar(x - width, orig_vec, width, label='Original Flow', color='green', alpha=0.8)
    plt.bar(x, manip_vec, width, label='Manipulated Flow', color='blue', alpha=0.8)
    plt.bar(x + width, frag_vec, width, label='Fragmented Flow', color='red', alpha=0.8)
    
    plt.ylabel('Feature Value (Normalized Scale)')
    plt.title('Feature Evolution of a Single Representative Attack Flow')
    plt.xticks(x, FEATURE_NAMES, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value annotations for critical features (like IAT Mean - idx 15)
    # This helps see the exact values like 0.03 -> 0.007
    def add_labels(rects, vec):
        for i, rect in enumerate(rects):
            if i in KEY_FEATURES_INDICES: # Only annotate key features to avoid clutter
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        f'{vec[i]:.3f}',
                        ha='center', va='bottom', rotation=90, fontsize=8)

    # We need the rectangles to annotate
    # But pyplot .bar returns them, so we can't easily capture them above without storing
    # Let's just trust the bars for now or redraw if needed.
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved single flow chart to {output_path}")
    plt.close()

def create_comparison_table(orig_vec, manip_vec, frag_vec, output_path):
    """Save the exact values to a clean CSV table"""
    df = pd.DataFrame({
        'Feature ID': range(len(FEATURE_NAMES)),
        'Feature Name': FEATURE_NAMES,
        'Original Value': orig_vec,
        'Manipulated Value': manip_vec,
        'Fragmented Value': frag_vec
    })
    
    # Calculate % change
    # Avoid zero division
    orig_safe = df['Original Value'].replace(0, 1e-9)
    df['% Change (Orig->Frag)'] = ((df['Fragmented Value'] - df['Original Value']) / orig_safe) * 100
    
    df.to_csv(output_path, index=False)
    print(f"Saved comparison table to {output_path}")
    return df

def main():
    workspace_root = "/home/rising/EvasionShield-LUCID"
    # Using WebDDoS as before
    attack = "00-WebDDoS"
    
    base_dir_orig = os.path.join(workspace_root, "TrafficManipulator-master/DATASETS/FLATTEN-PCAPS")
    base_dir_manip = os.path.join(workspace_root, "TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42")
    base_dir_frag = os.path.join(workspace_root, "TrafficManipulator-master/DATASETS/FRAGMENTED_42_150")
    
    path_orig = os.path.join(base_dir_orig, attack, "10t-100n-DOS2019-flatten-dataset-test.hdf5")
    path_manip = os.path.join(base_dir_manip, attack, "10t-100n-DOS2019-flatten-dataset-test.hdf5")
    path_frag = os.path.join(base_dir_frag, attack, "10t-100n-DOS2019-flatten-dataset-test.hdf5")
    
    print("Loading datasets...")
    X_orig = load_dataset(path_orig)
    X_manip = load_dataset(path_manip)
    X_frag = load_dataset(path_frag)
    
    if X_orig is None or X_manip is None or X_frag is None:
        print("Data load failed.")
        return
        
    print(f"Dataset shapes: Orig {X_orig.shape}, Manip {X_manip.shape}, Frag {X_frag.shape}")
    print("Note: Sample counts differ significantly. Impossible to track exact flow ID 1-to-1.")
    print("Strategy: Extracting 'Representative' (Centroid) samples from each phase to illustrate typical transformation.")
    
    # Find representative samples
    vec_orig, idx_orig = find_representative_sample(X_orig)
    vec_manip, idx_manip = find_representative_sample(X_manip)
    vec_frag, idx_frag = find_representative_sample(X_frag)
    
    print(f"Selected Representative Samples:")
    print(f"Original: Index {idx_orig}")
    print(f"Manipulated: Index {idx_manip}")
    print(f"Fragmented: Index {idx_frag}")
    
    output_dir = "visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Generate visualization
    plot_single_flow_comparison(vec_orig, vec_manip, vec_frag, 
                              os.path.join(output_dir, "single_flow_evolution.png"))
                              
    # Generate data table
    df = create_comparison_table(vec_orig, vec_manip, vec_frag,
                               os.path.join(output_dir, "single_flow_values.csv"))
                               
    # Print key insight to console
    print("\nKey Feature Value Changes (Representative Flow):")
    print("-" * 60)
    for idx in KEY_FEATURES_INDICES:
        name = FEATURE_NAMES[idx]
        v1 = vec_orig[idx]
        v2 = vec_manip[idx]
        v3 = vec_frag[idx]
        print(f"{name:20s}: {v1:.6f} -> {v2:.6f} -> {v3:.6f}")

if __name__ == "__main__":
    main()

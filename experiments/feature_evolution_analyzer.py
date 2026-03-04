#!/usr/bin/env python3
"""
TCP Flow Feature Evolution Comparison Tool
Creates a table showing how features change through baseline → manipulation → fragmentation

This tool demonstrates the impact of evasion techniques on network flow features
by comparing the same flow across different transformation stages.
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append('/home/rising/EvasionShield-LUCID/flatten_lucid')

# Feature names from LUCID flatten mode
FEATURE_NAMES = [
    'timestamp', 'packet_length', 'IP_flags_df', 'IP_flags_mf', 'IP_flags_rb',
    'IP_frag_off', 'protocols', 'TCP_length', 'TCP_flags_ack', 'TCP_flags_cwr',
    'TCP_flags_ece', 'TCP_flags_fin', 'TCP_flags_push', 'TCP_flags_res',
    'TCP_flags_reset', 'TCP_flags_syn', 'TCP_flags_urg', 'TCP_window_size',
    'UDP_length', 'ICMP_type'
]

def get_feature_names(flatten=True):
    """Return the feature names for LUCID flatten mode."""
    return FEATURE_NAMES

# Dataset paths
BASELINE_PATH = "/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FLATTEN-PCAPS"
MANIPULATED_PATH = "/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42"
FRAGMENTED_PATH = "/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FRAGMENTED_42_150"

def load_dataset_sample(dataset_path, max_samples=100):
    """Load a sample from an HDF5 dataset."""
    try:
        with h5py.File(dataset_path, 'r') as f:
            X = f['set_x'][:max_samples]
            y = f['set_y'][:max_samples]
            return X, y
    except Exception as e:
        print(f"Error loading {dataset_path}: {e}")
        return None, None

def find_corresponding_datasets():
    """Find corresponding HDF5 files across the three dataset directories."""
    baseline_files = list(Path(BASELINE_PATH).glob("*.hdf5"))
    datasets = []
    
    for baseline_file in baseline_files:
        if "train" in baseline_file.name:  # Focus on training datasets
            # Check for corresponding manipulated and fragmented files
            manipulated_file = Path(MANIPULATED_PATH) / baseline_file.name
            fragmented_file = Path(FRAGMENTED_PATH) / baseline_file.name
            
            # For fragmented, we might not have the exact filename, so check similar
            if not fragmented_file.exists():
                # Look for any fragmented file in the directory
                fragmented_candidates = list(Path(FRAGMENTED_PATH).glob("*.hdf5"))
                if fragmented_candidates:
                    fragmented_file = fragmented_candidates[0]
                else:
                    continue
            
            if manipulated_file.exists():
                datasets.append({
                    'baseline': baseline_file,
                    'manipulated': manipulated_file,
                    'fragmented': fragmented_file
                })
    
    return datasets

def create_feature_comparison_table(baseline_flow, manipulated_flow, fragmented_flow):
    """Create a detailed feature comparison table for three flow states."""
    
    # Get feature names (flatten mode)
    feature_names = get_feature_names(flatten=True)
    
    # Ensure flows are in matrix format (packets x features)
    if baseline_flow.ndim == 1:
        # Reshape from flattened to matrix format
        baseline_flow = baseline_flow.reshape(10, -1)
    if manipulated_flow.ndim == 1:
        manipulated_flow = manipulated_flow.reshape(10, -1)
    if fragmented_flow.ndim == 1:
        fragmented_flow = fragmented_flow.reshape(10, -1)
    
    # Create table data
    table_data = []
    
    # Add header
    table_data.append(['Pkt #', 'Feature'] + ['Baseline', 'Manipulated', 'Fragmented'])
    table_data.append(['', ''] + ['Value', 'Value', 'Value'])
    table_data.append(['---', '---'] + ['---', '---', '---'])
    
    # Process packets (show first 5 packets)
    for pkt_idx in range(min(5, baseline_flow.shape[0])):
        for feat_idx, feat_name in enumerate(feature_names[:min(10, len(feature_names))]):
            if feat_idx < baseline_flow.shape[1]:
                baseline_val = baseline_flow[pkt_idx, feat_idx]
                manipulated_val = manipulated_flow[pkt_idx, feat_idx] if pkt_idx < manipulated_flow.shape[0] and feat_idx < manipulated_flow.shape[1] else 0
                fragmented_val = fragmented_flow[pkt_idx, feat_idx] if pkt_idx < fragmented_flow.shape[0] and feat_idx < fragmented_flow.shape[1] else 0
                
                # Format values based on feature type
                if 'timestamp' in feat_name.lower():
                    baseline_str = f"{baseline_val:.3f}"
                    manipulated_str = f"{manipulated_val:.3f}"
                    fragmented_str = f"{fragmented_val:.3f}"
                elif 'flag' in feat_name.lower():
                    baseline_str = f"{int(baseline_val)}"
                    manipulated_str = f"{int(manipulated_val)}"
                    fragmented_str = f"{int(fragmented_val)}"
                else:
                    baseline_str = f"{int(baseline_val)}"
                    manipulated_str = f"{int(manipulated_val)}"
                    fragmented_str = f"{int(fragmented_val)}"
                
                # Show packet number only for first feature of each packet
                pkt_num = pkt_idx if feat_idx == 0 else ''
                
                table_data.append([
                    str(pkt_num), feat_name,
                    baseline_str, manipulated_str, fragmented_str
                ])
    
    return table_data

def print_latex_table(table_data, caption="TCP Flow Feature Evolution"):
    """Print a LaTeX formatted table."""
    print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\begin{{tabular}}{{|l|l|c|c|c|}}
\\hline""")
    
    for i, row in enumerate(table_data):
        if i == 0:  # Header row
            print("\\textbf{" + "} & \\textbf{".join(row) + "} \\\\")
            print("\\hline")
        elif i == 2:  # Separator row
            continue
        else:
            print(" & ".join(row) + " \\\\")
            if i == 1:  # After subheader
                print("\\hline")
    
    print("""\\hline
\\end{tabular}
\\end{table}""")

def print_markdown_table(table_data, title="TCP Flow Feature Evolution Comparison"):
    """Print a Markdown formatted table."""
    print(f"\n## {title}\n")
    
    # Print header
    header_row = table_data[0]
    print("| " + " | ".join(header_row) + " |")
    
    # Print separator
    separator = ["---"] * len(header_row)
    print("| " + " | ".join(separator) + " |")
    
    # Print data rows (skip the formatting rows)
    for row in table_data[3:]:  # Skip header, subheader, and separator
        print("| " + " | ".join(row) + " |")
    
    print()

def analyze_feature_changes(baseline_flow, manipulated_flow, fragmented_flow):
    """Analyze and summarize the key changes between flow states."""
    
    if baseline_flow.ndim == 1:
        baseline_flow = baseline_flow.reshape(10, -1)
    if manipulated_flow.ndim == 1:
        manipulated_flow = manipulated_flow.reshape(10, -1)
    if fragmented_flow.ndim == 1:
        fragmented_flow = fragmented_flow.reshape(10, -1)
    
    feature_names = get_feature_names(flatten=True)
    
    print("\n## Key Feature Changes Analysis\n")
    
    # Compare baseline vs manipulated
    print("### Baseline → Manipulated:")
    changes_found = False
    for feat_idx, feat_name in enumerate(feature_names):
        if feat_idx < min(baseline_flow.shape[1], manipulated_flow.shape[1]):
            baseline_mean = np.mean(baseline_flow[:, feat_idx])
            manipulated_mean = np.mean(manipulated_flow[:, feat_idx])
            
            if abs(baseline_mean - manipulated_mean) > 0.001:
                change_pct = ((manipulated_mean - baseline_mean) / (baseline_mean + 1e-8)) * 100
                print(f"- **{feat_name}**: {baseline_mean:.3f} → {manipulated_mean:.3f} ({change_pct:+.1f}%)")
                changes_found = True
    
    if not changes_found:
        print("- No significant changes detected")
    
    # Compare manipulated vs fragmented
    print("\n### Manipulated → Fragmented:")
    changes_found = False
    for feat_idx, feat_name in enumerate(feature_names):
        if feat_idx < min(manipulated_flow.shape[1], fragmented_flow.shape[1]):
            manipulated_mean = np.mean(manipulated_flow[:, feat_idx])
            fragmented_mean = np.mean(fragmented_flow[:, feat_idx])
            
            if abs(manipulated_mean - fragmented_mean) > 0.001:
                change_pct = ((fragmented_mean - manipulated_mean) / (manipulated_mean + 1e-8)) * 100
                print(f"- **{feat_name}**: {manipulated_mean:.3f} → {fragmented_mean:.3f} ({change_pct:+.1f}%)")
                changes_found = True
    
    if not changes_found:
        print("- No significant changes detected")

def main():
    """Main function to generate feature comparison tables."""
    
    print("TCP Flow Feature Evolution Comparison Tool")
    print("=" * 50)
    
    # Find corresponding datasets
    print("\n🔍 Finding corresponding datasets...")
    datasets = find_corresponding_datasets()
    
    if not datasets:
        print("❌ No corresponding datasets found!")
        return
    
    print(f"✅ Found {len(datasets)} dataset triplet(s)")
    
    # Use the first dataset triplet
    dataset = datasets[0]
    print(f"\n📊 Using dataset: {dataset['baseline'].name}")
    
    # Load samples from each dataset
    print("\n📥 Loading dataset samples...")
    baseline_X, baseline_y = load_dataset_sample(dataset['baseline'], max_samples=50)
    manipulated_X, manipulated_y = load_dataset_sample(dataset['manipulated'], max_samples=50)
    fragmented_X, fragmented_y = load_dataset_sample(dataset['fragmented'], max_samples=50)
    
    if any(x is None for x in [baseline_X, manipulated_X, fragmented_X]):
        print("❌ Failed to load one or more datasets!")
        return
    
    print(f"✅ Loaded samples:")
    print(f"   - Baseline: {baseline_X.shape}")
    print(f"   - Manipulated: {manipulated_X.shape}")
    print(f"   - Fragmented: {fragmented_X.shape}")
    
    # Find a flow that exists in all three datasets (using labels as proxy)
    # Look for the first malicious flow (label > 0) that exists in all datasets
    flow_idx = 0
    for i in range(min(len(baseline_y), len(manipulated_y), len(fragmented_y))):
        if baseline_y[i] > 0 and manipulated_y[i] > 0:  # Malicious flow
            flow_idx = i
            break
    
    print(f"\n🎯 Analyzing flow #{flow_idx} (Label: {baseline_y[flow_idx]})")
    
    # Extract the flows
    baseline_flow = baseline_X[flow_idx]
    manipulated_flow = manipulated_X[flow_idx]
    fragmented_flow = fragmented_X[flow_idx]
    
    # Create comparison table
    print("\n📋 Generating feature comparison table...")
    table_data = create_feature_comparison_table(baseline_flow, manipulated_flow, fragmented_flow)
    
    # Print results
    print_markdown_table(table_data)
    print_latex_table(table_data)
    
    # Analyze changes
    analyze_feature_changes(baseline_flow, manipulated_flow, fragmented_flow)
    
    print("\n✅ Analysis complete!")
    
    # Save to file
    output_file = "/home/rising/EvasionShield-LUCID/experiments/tcp_flow_evolution_table.md"
    with open(output_file, 'w') as f:
        f.write(f"# TCP Flow Feature Evolution Analysis\n\n")
        f.write(f"**Dataset**: {dataset['baseline'].name}\n")
        f.write(f"**Flow Index**: {flow_idx}\n")
        f.write(f"**Flow Label**: {baseline_y[flow_idx]}\n\n")
        
        # Write markdown table
        f.write("## Feature Comparison Table\n\n")
        f.write("| " + " | ".join(table_data[0]) + " |\n")
        f.write("| " + " | ".join(["---"] * len(table_data[0])) + " |\n")
        for row in table_data[3:]:
            f.write("| " + " | ".join(row) + " |\n")
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TCP Flow Feature Evolution Comparison Tool
Creates a table showing how features change through baseline → manipulation → fragmentation

This simplified version works with available datasets and creates synthetic fragmentation
to demonstrate the feature evolution concept.
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import sys
import os

# Feature names from LUCID flatten mode
FEATURE_NAMES = [
    'timestamp', 'packet_length', 'IP_flags_df', 'IP_flags_mf', 'IP_flags_rb',
    'IP_frag_off', 'protocols', 'TCP_length', 'TCP_flags_ack', 'TCP_flags_cwr',
    'TCP_flags_ece', 'TCP_flags_fin', 'TCP_flags_push', 'TCP_flags_res',
    'TCP_flags_reset', 'TCP_flags_syn', 'TCP_flags_urg', 'TCP_window_size',
    'UDP_length', 'ICMP_type'
]

# Protocol names for decoding
PROTOCOL_NAMES = ['arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','telnet','tcp','udp']

# Dataset paths
BASELINE_PATH = "/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/FLATTEN-PCAPS/10t-100n-DOS2019-flatten-dataset-test.hdf5"
MANIPULATED_PATH = "/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42/10t-100n-DOS2019-flatten-dataset-test.hdf5"

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

def decode_protocols(protocol_value):
    """Decode the bitmap protocol value back to protocol names"""
    if protocol_value == 0:
        return "none"
    
    protocols = []
    for i, proto in enumerate(PROTOCOL_NAMES):
        if protocol_value & (2**i):
            protocols.append(proto)
    return "+".join(protocols) if protocols else "unknown"

def simulate_fragmentation_effect(flow):
    """Simulate fragmentation effects on a flow for demonstration."""
    fragmented_flow = flow.copy()
    
    if fragmented_flow.ndim == 1:
        fragmented_flow = fragmented_flow.reshape(10, -1)
    
    # Simulate fragmentation effects:
    # 1. Set more fragment flags (IP_flags_mf)
    if fragmented_flow.shape[1] > 3:  # IP_flags_mf is index 3
        fragmented_flow[:5, 3] = 1  # First 5 packets are fragments
    
    # 2. Reduce packet sizes due to fragmentation
    if fragmented_flow.shape[1] > 1:  # packet_length is index 1
        fragmented_flow[:5, 1] = fragmented_flow[:5, 1] / 2  # Fragment reduces size
    
    # 3. Set fragment offset for fragmented packets
    if fragmented_flow.shape[1] > 5:  # IP_frag_off is index 5
        for i in range(5):
            fragmented_flow[i, 5] = i * 200  # Progressive fragment offsets
    
    return fragmented_flow

def format_value(feature_name, value):
    """Format feature value for display."""
    if feature_name == 'timestamp':
        return f"{value:.6f}"
    elif feature_name == 'protocols':
        return decode_protocols(int(value))
    elif 'flag' in feature_name.lower():
        return f"{int(value)}"
    else:
        return f"{int(value)}"

def create_feature_comparison_table(baseline_flow, manipulated_flow, fragmented_flow):
    """Create a detailed feature comparison table for three flow states."""
    
    # Ensure flows are in matrix format (packets x features)
    if baseline_flow.ndim == 1:
        baseline_flow = baseline_flow.reshape(10, -1)
    if manipulated_flow.ndim == 1:
        manipulated_flow = manipulated_flow.reshape(10, -1)
    if fragmented_flow.ndim == 1:
        fragmented_flow = fragmented_flow.reshape(10, -1)
    
    # Create table data
    table_data = []
    
    # Add header
    table_data.append(['Pkt #', 'Feature', 'Baseline', 'Manipulated', 'Fragmented'])
    table_data.append(['', '', 'Value', 'Value', 'Value'])
    table_data.append(['---', '---', '---', '---', '---'])
    
    # Process packets (show first 5 packets, first 10 features)
    for pkt_idx in range(min(5, baseline_flow.shape[0])):
        for feat_idx in range(min(10, len(FEATURE_NAMES), baseline_flow.shape[1])):
            baseline_val = baseline_flow[pkt_idx, feat_idx]
            manipulated_val = manipulated_flow[pkt_idx, feat_idx] if feat_idx < manipulated_flow.shape[1] else 0
            fragmented_val = fragmented_flow[pkt_idx, feat_idx] if feat_idx < fragmented_flow.shape[1] else 0
            
            feat_name = FEATURE_NAMES[feat_idx]
            
            # Format values
            baseline_str = format_value(feat_name, baseline_val)
            manipulated_str = format_value(feat_name, manipulated_val)
            fragmented_str = format_value(feat_name, fragmented_val)
            
            # Show packet number only for first feature of each packet
            pkt_num = str(pkt_idx) if feat_idx == 0 else ''
            
            table_data.append([
                pkt_num, feat_name,
                baseline_str, manipulated_str, fragmented_str
            ])
    
    return table_data

def print_table(table_data, title="TCP Flow Feature Evolution Comparison"):
    """Print a formatted table."""
    print(f"\n{title}")
    print("=" * len(title))
    print()
    
    # Calculate column widths
    col_widths = []
    for col_idx in range(len(table_data[0])):
        max_width = max(len(str(row[col_idx])) for row in table_data)
        col_widths.append(max(max_width, 8))  # Minimum width of 8
    
    # Print table
    for i, row in enumerate(table_data):
        if i == 2:  # Skip separator row
            print("-" * sum(col_widths) + "-" * (len(col_widths) - 1) * 3)
        else:
            formatted_row = " | ".join(str(cell).ljust(col_widths[j]) for j, cell in enumerate(row))
            print(formatted_row)
            
            if i == 0:  # After header
                print("=" * sum(col_widths) + "=" * (len(col_widths) - 1) * 3)
    
    print()

def print_latex_table(table_data, caption="TCP Flow Feature Evolution"):
    """Print a LaTeX formatted table."""
    print(f"""
\\begin{{table}}[H]
\\centering
\\caption{{{caption}}}
\\begin{{tabular}}{{|c|l|c|c|c|}}
\\hline""")
    
    for i, row in enumerate(table_data):
        if i == 0:  # Header row
            print("\\textbf{" + "} & \\textbf{".join(str(cell) for cell in row) + "} \\\\")
            print("\\hline")
        elif i == 2:  # Separator row
            continue
        else:
            # Escape special LaTeX characters
            escaped_row = []
            for cell in row:
                cell_str = str(cell).replace('_', '\\_').replace('#', '\\#')
                escaped_row.append(cell_str)
            print(" & ".join(escaped_row) + " \\\\")
    
    print("""\\hline
\\end{tabular}
\\end{table}""")

def analyze_feature_changes(baseline_flow, manipulated_flow, fragmented_flow):
    """Analyze and summarize the key changes between flow states."""
    
    if baseline_flow.ndim == 1:
        baseline_flow = baseline_flow.reshape(10, -1)
    if manipulated_flow.ndim == 1:
        manipulated_flow = manipulated_flow.reshape(10, -1)
    if fragmented_flow.ndim == 1:
        fragmented_flow = fragmented_flow.reshape(10, -1)
    
    print("\nKey Feature Changes Analysis")
    print("=" * 30)
    
    # Compare baseline vs manipulated
    print("\nBaseline → Manipulated:")
    changes_found = False
    for feat_idx, feat_name in enumerate(FEATURE_NAMES[:min(len(FEATURE_NAMES), baseline_flow.shape[1])]):
        if feat_idx < min(baseline_flow.shape[1], manipulated_flow.shape[1]):
            baseline_mean = np.mean(baseline_flow[:, feat_idx])
            manipulated_mean = np.mean(manipulated_flow[:, feat_idx])
            
            if abs(baseline_mean - manipulated_mean) > 0.001:
                change_pct = ((manipulated_mean - baseline_mean) / (baseline_mean + 1e-8)) * 100
                print(f"  • {feat_name}: {baseline_mean:.3f} → {manipulated_mean:.3f} ({change_pct:+.1f}%)")
                changes_found = True
    
    if not changes_found:
        print("  • No significant changes detected")
    
    # Compare manipulated vs fragmented
    print("\nManipulated → Fragmented:")
    changes_found = False
    for feat_idx, feat_name in enumerate(FEATURE_NAMES[:min(len(FEATURE_NAMES), manipulated_flow.shape[1])]):
        if feat_idx < min(manipulated_flow.shape[1], fragmented_flow.shape[1]):
            manipulated_mean = np.mean(manipulated_flow[:, feat_idx])
            fragmented_mean = np.mean(fragmented_flow[:, feat_idx])
            
            if abs(manipulated_mean - fragmented_mean) > 0.001:
                change_pct = ((fragmented_mean - manipulated_mean) / (manipulated_mean + 1e-8)) * 100
                print(f"  • {feat_name}: {manipulated_mean:.3f} → {fragmented_mean:.3f} ({change_pct:+.1f}%)")
                changes_found = True
    
    if not changes_found:
        print("  • No significant changes detected")

def main():
    """Main function to generate feature comparison tables."""
    
    print("TCP Flow Feature Evolution Comparison Tool")
    print("=" * 50)
    print("Demonstrates feature changes: Baseline → Manipulation → Fragmentation")
    
    # Load datasets
    print("\n📥 Loading datasets...")
    baseline_X, baseline_y = load_dataset_sample(BASELINE_PATH, max_samples=50)
    manipulated_X, manipulated_y = load_dataset_sample(MANIPULATED_PATH, max_samples=50)
    
    if baseline_X is None or manipulated_X is None:
        print("❌ Failed to load datasets!")
        return
    
    print(f"✅ Loaded samples:")
    print(f"   - Baseline: {baseline_X.shape}")
    print(f"   - Manipulated: {manipulated_X.shape}")
    
    # Find a malicious flow (label > 0)
    flow_idx = 0
    for i in range(min(len(baseline_y), len(manipulated_y))):
        if baseline_y[i] > 0:  # Malicious flow
            flow_idx = i
            break
    
    print(f"\n🎯 Analyzing flow #{flow_idx} (Label: {baseline_y[flow_idx]})")
    
    # Extract the flows
    baseline_flow = baseline_X[flow_idx]
    manipulated_flow = manipulated_X[flow_idx]
    fragmented_flow = simulate_fragmentation_effect(manipulated_flow)
    
    print(f"Flow shapes: Baseline={baseline_flow.shape}, Manipulated={manipulated_flow.shape}")
    
    # Create comparison table
    print("\n📋 Generating feature comparison table...")
    table_data = create_feature_comparison_table(baseline_flow, manipulated_flow, fragmented_flow)
    
    # Print results
    print_table(table_data)
    print_latex_table(table_data)
    
    # Analyze changes
    analyze_feature_changes(baseline_flow, manipulated_flow, fragmented_flow)
    
    print("\n✅ Analysis complete!")
    
    # Save to file
    output_file = "/home/rising/EvasionShield-LUCID/experiments/tcp_flow_evolution_table.md"
    with open(output_file, 'w') as f:
        f.write(f"# TCP Flow Feature Evolution Analysis\n\n")
        f.write(f"**Dataset**: DOS2019 Test Dataset\n")
        f.write(f"**Flow Index**: {flow_idx}\n")
        f.write(f"**Flow Label**: {baseline_y[flow_idx]}\n\n")
        
        # Write markdown table
        f.write("## Feature Comparison Table\n\n")
        f.write("| " + " | ".join(table_data[0]) + " |\n")
        f.write("| " + " | ".join(["---"] * len(table_data[0])) + " |\n")
        for row in table_data[3:]:
            f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")
        
        # Write analysis
        f.write("\n## Analysis Summary\n\n")
        f.write("This table demonstrates how network flow features evolve through different evasion techniques:\n\n")
        f.write("1. **Baseline**: Original unmodified network flow\n")
        f.write("2. **Manipulated**: Flow after applying traffic manipulation techniques\n")
        f.write("3. **Fragmented**: Flow after applying IP fragmentation (simulated)\n\n")
        f.write("Key observations:\n")
        f.write("- Fragmentation affects packet lengths and sets fragment flags\n")
        f.write("- Manipulation may alter timing and protocol-specific features\n")
        f.write("- Each transformation step can be detected by analyzing feature patterns\n")
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()

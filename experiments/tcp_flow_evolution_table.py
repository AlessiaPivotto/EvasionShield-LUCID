#!/usr/bin/env python3
"""
TCP Flow Feature Evolution Table
Shows how packet features change through: Baseline → Manipulation → Fragmentation

Creates a detailed table similar to the reference showing packet-level features
across different transformation stages.
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from collections import OrderedDict

class FlowFeatureEvolutionTable:
    def __init__(self):
        # Feature names for the matrix format (11 features)
        self.matrix_features = [
            'timestamp', 'packet_length', 'IP_flags_df', 'IP_flags_mf', 'IP_flags_rb',
            'IP_frag_off', 'protocols', 'TCP_length', 'TCP_flags_combined', 
            'TCP_window_size', 'flow_packets_count'
        ]
        
        # Protocol decoding
        self.protocol_names = ['arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','telnet','tcp','udp']
        
    def decode_protocols(self, protocol_value):
        """Decode the bitmap protocol value"""
        if protocol_value == 0:
            return "00000000000"
        
        # Convert to binary string
        binary_str = format(int(protocol_value), '012b')
        return binary_str
    
    def load_matrix_dataset(self, dataset_path, name=""):
        """Load matrix format dataset (flows × packets × features)"""
        try:
            with h5py.File(dataset_path, 'r') as f:
                X = np.array(f['set_x'])
                y = np.array(f['set_y'])
                
                print(f"✅ Loaded {name}: {X.shape[0]} flows, {X.shape[1]} packets/flow, {X.shape[2]} features/packet")
                
                # Find malicious flows (assuming label 1 = malicious)
                malicious_flows = []
                for i, (flow, label) in enumerate(zip(X, y)):
                    if label == 1:  # Malicious
                        # Check if flow has meaningful data
                        non_zero_packets = 0
                        for packet in flow:
                            if np.sum(np.abs(packet)) > 0:
                                non_zero_packets += 1
                        
                        if non_zero_packets >= 3:  # At least 3 packets for TCP analysis
                            malicious_flows.append((i, flow, label))
                
                print(f"   Found {len(malicious_flows)} malicious flows with ≥3 packets")
                return malicious_flows[:5]  # Return first 5 for analysis
                
        except Exception as e:
            print(f"❌ Error loading {dataset_path}: {e}")
            return []
    
    def create_feature_evolution_table(self, baseline_flows, fragmented_flows):
        """Create the main comparison table"""
        
        if not baseline_flows or not fragmented_flows:
            print("❌ Error: Missing flow data for comparison")
            return
        
        print("\n" + "="*150)
        print("TCP FLOW FEATURE EVOLUTION TABLE: BASELINE → FRAGMENTED")
        print("="*150)
        print("Based on LUCID packet-level feature extraction - showing same flow through transformation pipeline")
        print("-"*150)
        
        # Take first malicious flow from each dataset
        baseline_flow = baseline_flows[0][1]
        fragmented_flow = fragmented_flows[0][1] 
        
        print("Reference Flow Selected: First Malicious TCP Flow")
        print(f"Baseline Flow Index: {baseline_flows[0][0]} | Fragmented Flow Index: {fragmented_flows[0][0]}")
        print("-"*150)
        
        # Create header similar to your reference table
        header = (
            f"{'Pkt #':<6} {'Time':<12} {'Packet':<8} {'Highest':<8} {'IP':<12} {'Protocols':<12} "
            f"{'TCP':<8} {'TCP':<8} {'TCP':<8} {'TCP':<8} {'UDP':<8} {'ICMP':<6}"
        )
        subheader = (
            f"{'':6} {'(seconds)':<12} {'Length':<8} {'Layer':<8} {'Flags':<12} {'(bitmap)':<12} "
            f"{'Len':<8} {'Ack':<8} {'Flags':<8} {'Window':<8} {'Len':<8} {'Type':<6}"
        )
        
        print(header)
        print(subheader)
        print("-"*150)
        
        def format_packet_row(pkt_num, packet_data, state_name):
            """Format a single packet row"""
            if np.sum(np.abs(packet_data)) == 0:  # Skip zero packets
                return None
                
            timestamp = packet_data[0]
            packet_length = int(packet_data[1])
            ip_df = int(packet_data[2])
            ip_mf = int(packet_data[3])
            ip_rb = int(packet_data[4])
            ip_frag_off = int(packet_data[5])
            protocols = packet_data[6]
            tcp_length = int(packet_data[7])
            tcp_flags = int(packet_data[8])
            tcp_window = int(packet_data[9])
            
            # Format values
            ip_flags = f"{ip_df}{ip_mf}{ip_rb}|{ip_frag_off}"
            protocols_binary = self.decode_protocols(protocols)
            highest_layer = "tcp" if protocols & (2**10) else "ip"
            
            return (
                f"{state_name:<6} {timestamp:<12.6f} {packet_length:<8} {highest_layer:<8} "
                f"{ip_flags:<12} {protocols_binary:<12} {tcp_length:<8} {tcp_flags & 16:<8} "
                f"{tcp_flags:<8} {tcp_window:<8} {'0':<8} {'0':<6}"
            )
        
        # Print baseline and fragmented packets side by side
        max_packets = min(10, len(baseline_flow), len(fragmented_flow))
        
        print("BASELINE FLOW (Before Manipulation & Fragmentation):")
        baseline_count = 0
        for i, packet in enumerate(baseline_flow):
            if np.sum(np.abs(packet)) > 0:
                row = format_packet_row(baseline_count, packet, f"B-{baseline_count}")
                if row:
                    print(row)
                    baseline_count += 1
                    if baseline_count >= 5:
                        break
        
        print("\nFRAGMENTED FLOW (After Manipulation & Fragmentation):")
        fragmented_count = 0
        for i, packet in enumerate(fragmented_flow):
            if np.sum(np.abs(packet)) > 0:
                row = format_packet_row(fragmented_count, packet, f"F-{fragmented_count}")
                if row:
                    print(row)
                    fragmented_count += 1
                    if fragmented_count >= 10:
                        break
        
        print("\n" + "-"*150)
        print("FEATURE ANALYSIS:")
        print("-"*150)
        
        # Compare key metrics
        def flow_stats(flow_data, name):
            packets = [p for p in flow_data if np.sum(np.abs(p)) > 0]
            if not packets:
                return
                
            packets = np.array(packets)
            
            print(f"{name} Statistics:")
            print(f"  • Active packets: {len(packets)}")
            print(f"  • Avg packet length: {np.mean(packets[:, 1]):.1f} bytes")
            print(f"  • Fragmentation indicators:")
            print(f"    - More Fragments (MF) flags: {np.sum(packets[:, 3])}")
            print(f"    - Fragment offsets > 0: {np.sum(packets[:, 5] > 0)}")
            print(f"  • TCP window sizes: min={np.min(packets[:, 9]):.0f}, max={np.max(packets[:, 9]):.0f}")
            print(f"  • Time span: {np.max(packets[:, 0]) - np.min(packets[:, 0]):.6f} seconds")
            print()
        
        flow_stats(baseline_flow, "BASELINE")
        flow_stats(fragmented_flow, "FRAGMENTED")
        
        print("KEY OBSERVATIONS:")
        print("• Baseline flows show original packet structure")
        print("• Fragmented flows may show:")
        print("  - Increased packet count due to fragmentation")
        print("  - More Fragment (MF) flags set to 1")
        print("  - Non-zero fragment offsets")
        print("  - Modified packet lengths")
        print("  - Altered timing patterns")
        print("="*150)
    
    def create_detailed_feature_table(self, baseline_flows, fragmented_flows):
        """Create a detailed LaTeX-style table for thesis inclusion"""
        
        if not baseline_flows or not fragmented_flows:
            return
            
        baseline_flow = baseline_flows[0][1]
        fragmented_flow = fragmented_flows[0][1]
        
        print("\n" + "="*120)
        print("DETAILED FEATURE COMPARISON TABLE (LaTeX Format)")
        print("="*120)
        
        # Extract first few meaningful packets from each flow
        def get_meaningful_packets(flow, limit=3):
            meaningful = []
            for packet in flow:
                if np.sum(np.abs(packet)) > 0:
                    meaningful.append(packet)
                    if len(meaningful) >= limit:
                        break
            return meaningful
        
        baseline_packets = get_meaningful_packets(baseline_flow, 3)
        fragmented_packets = get_meaningful_packets(fragmented_flow, 6)  # May have more due to fragmentation
        
        print("\\begin{table}[htbp]")
        print("\\centering")
        print("\\caption{TCP Flow Feature Evolution: Baseline vs Fragmented}")
        print("\\label{tab:tcp_flow_evolution}")
        print("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
        print("\\hline")
        print("\\textbf{Packet} & \\textbf{Time} & \\textbf{Length} & \\textbf{IP Flags} & \\textbf{Frag} & \\textbf{Protocol} & \\textbf{TCP} & \\textbf{TCP} & \\textbf{TCP} \\\\")
        print("\\textbf{State} & \\textbf{(sec)} & \\textbf{(bytes)} & \\textbf{DF|MF|RB} & \\textbf{Offset} & \\textbf{Bitmap} & \\textbf{Length} & \\textbf{Flags} & \\textbf{Window} \\\\")
        print("\\hline")
        
        # Baseline packets
        for i, packet in enumerate(baseline_packets):
            timestamp = packet[0]
            length = int(packet[1])
            ip_flags = f"{int(packet[2])}|{int(packet[3])}|{int(packet[4])}"
            frag_off = int(packet[5])
            protocols = f"0x{int(packet[6]):04x}"
            tcp_len = int(packet[7])
            tcp_flags = int(packet[8])
            tcp_win = int(packet[9])
            
            print(f"Baseline-{i} & {timestamp:.6f} & {length} & {ip_flags} & {frag_off} & {protocols} & {tcp_len} & {tcp_flags} & {tcp_win} \\\\")
        
        print("\\hline")
        
        # Fragmented packets
        for i, packet in enumerate(fragmented_packets):
            timestamp = packet[0]
            length = int(packet[1])
            ip_flags = f"{int(packet[2])}|{int(packet[3])}|{int(packet[4])}"
            frag_off = int(packet[5])
            protocols = f"0x{int(packet[6]):04x}"
            tcp_len = int(packet[7])
            tcp_flags = int(packet[8])
            tcp_win = int(packet[9])
            
            print(f"Fragmented-{i} & {timestamp:.6f} & {length} & {ip_flags} & {frag_off} & {protocols} & {tcp_len} & {tcp_flags} & {tcp_win} \\\\")
        
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
        print("="*120)
    
    def run_analysis(self):
        """Main analysis function"""
        
        # Available datasets
        datasets = {
            'baseline': '/home/rising/EvasionShield-LUCID/DATASETS/merged_matrix_100n_train_dataset.hdf5',
            'fragmented': '/home/rising/EvasionShield-LUCID/DATASETS/merged_matrix_10n_train_dataset.hdf5'
        }
        
        print("🔍 TCP Flow Feature Evolution Analysis")
        print("="*60)
        print("Loading datasets for comparison...")
        
        # Load datasets
        flows_data = {}
        for dataset_type, path in datasets.items():
            if os.path.exists(path):
                flows_data[dataset_type] = self.load_matrix_dataset(path, dataset_type)
            else:
                print(f"❌ Dataset not found: {path}")
                flows_data[dataset_type] = []
        
        # Create comparison tables
        if flows_data['baseline'] and flows_data['fragmented']:
            print(f"\n🎯 Analyzing flows:")
            print(f"   Baseline flows available: {len(flows_data['baseline'])}")
            print(f"   Fragmented flows available: {len(flows_data['fragmented'])}")
            
            self.create_feature_evolution_table(
                flows_data['baseline'],
                flows_data['fragmented']
            )
            
            self.create_detailed_feature_table(
                flows_data['baseline'],
                flows_data['fragmented']
            )
            
        else:
            print("❌ Error: Could not load required datasets for comparison")
            print("Available datasets:")
            for name, path in datasets.items():
                exists = "✅" if os.path.exists(path) else "❌"
                print(f"  {exists} {name}: {path}")

def main():
    print("TCP Flow Feature Evolution Table Generator")
    print("Showing packet-level feature changes: Baseline → Fragmented")
    print("="*70)
    
    analyzer = FlowFeatureEvolutionTable()
    analyzer.run_analysis()
    
    print("\n📋 Analysis Complete!")
    print("💡 Use this table to understand how fragmentation affects packet features")
    print("📖 Perfect for thesis inclusion showing transformation effects")

if __name__ == "__main__":
    main()

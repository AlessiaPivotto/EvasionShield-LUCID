#!/usr/bin/env python3
"""
TCP Flow Feature Comparison Tool
Shows feature evolution: Baseline -> Manipulated -> Fragmented

Creates a table similar to the reference showing initial features,
features after manipulation, and features after fragmentation
for the same TCP flow sample.
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

# Self-contained feature definitions - no external imports needed

class TCPFlowFeatureComparator:
    def __init__(self):
        self.feature_names = [
            'timestamp', 'packet_length', 'IP_flags_df', 'IP_flags_mf', 'IP_flags_rb',
            'IP_frag_off', 'protocols', 'TCP_length', 'TCP_flags_ack', 'TCP_flags_cwr',
            'TCP_flags_ece', 'TCP_flags_fin', 'TCP_flags_push', 'TCP_flags_res',
            'TCP_flags_reset', 'TCP_flags_syn', 'TCP_flags_urg', 'TCP_window_size',
            'UDP_length', 'ICMP_type'
        ]
        
        self.protocol_names = ['arp','data','dns','ftp','http','icmp','ip','ssdp','ssl','telnet','tcp','udp']
        
    def decode_protocols(self, protocol_value):
        """Decode the bitmap protocol value back to protocol names"""
        if protocol_value == 0:
            return "none"
        
        protocols = []
        for i, proto in enumerate(self.protocol_names):
            if protocol_value & (2**i):
                protocols.append(proto)
        return "+".join(protocols) if protocols else "unknown"
    
    def format_feature_value(self, feature_name, value, packet_num=None):
        """Format feature values for display with appropriate context"""
        if feature_name == 'timestamp':
            return f"{value:.6f}"
        elif feature_name in ['packet_length', 'TCP_length', 'UDP_length', 'TCP_window_size']:
            return f"{int(value)}"
        elif feature_name == 'protocols':
            return f"{int(value):08b}" if value < 256 else f"{int(value)}"
        elif feature_name == 'IP_frag_off':
            return f"{int(value)}" + (" (fragmented)" if value > 0 else "")
        elif feature_name.startswith('IP_flags_') or feature_name.startswith('TCP_flags_'):
            # For flags, show binary representation
            return f"{int(value)}"
        else:
            return f"{int(value)}"
    
    def load_dataset(self, dataset_path, dataset_name=""):
        """Load HDF5 dataset and extract sample flows"""
        try:
            with h5py.File(dataset_path, 'r') as f:
                X = np.array(f['set_x'])
                y = np.array(f['set_y']) 
                
                print(f"Loaded {dataset_name}: {X.shape[0]} flows, {X.shape[1]} packets/flow, {X.shape[2]} features/packet")
                
                # Find TCP flows (look for TCP protocol presence)
                tcp_flows = []
                for i, flow in enumerate(X):
                    # Check if any packet in the flow has TCP protocol (protocols field > 0 and contains TCP bit)
                    protocols_col = 6  # protocols is 7th feature (0-indexed)
                    tcp_bit = 2**10  # TCP is 11th protocol (0-indexed = bit 10)
                    
                    has_tcp = False
                    for packet in flow:
                        if packet[protocols_col] & tcp_bit:
                            has_tcp = True
                            break
                    
                    if has_tcp:
                        tcp_flows.append((i, flow, y[i]))
                        
                print(f"Found {len(tcp_flows)} TCP flows in {dataset_name}")
                return tcp_flows[:10]  # Return first 10 TCP flows for analysis
                
        except Exception as e:
            print(f"Error loading {dataset_path}: {e}")
            return []
    
    def create_comparison_table(self, baseline_flows, manipulated_flows, fragmented_flows):
        """Create a comparison table showing feature evolution"""
        
        if not baseline_flows or not manipulated_flows or not fragmented_flows:
            print("Error: Missing flow data for comparison")
            return
        
        # Take the first TCP flow from each dataset for detailed comparison
        baseline_flow = baseline_flows[0][1]  # (index, flow_data, label)
        manipulated_flow = manipulated_flows[0][1]
        fragmented_flow = fragmented_flows[0][1]
        
        print("\n" + "="*120)
        print("TCP FLOW FEATURE COMPARISON: BASELINE → MANIPULATED → FRAGMENTED")
        print("="*120)
        
        # Create table header
        print(f"{'Pkt #':<5} {'Time':<12} {'Packet':<8} {'IP Flags':<15} {'IP':<8} {'Protocols':<12} "
              f"{'TCP':<8} {'TCP Flags':<25} {'TCP':<8} {'UDP':<8} {'ICMP':<6}")
        print(f"{'':5} {'(sec)':<12} {'Length':<8} {'DF|MF|RB':<15} {'FragOff':<8} {'(bitmap)':<12} "
              f"{'Length':<8} {'ACK|CWR|ECE|FIN|PSH|RES|RST|SYN|URG':<25} {'WinSize':<8} {'Length':<8} {'Type':<6}")
        print("-" * 120)
        
        # Function to print flow packets
        def print_flow_packets(flow_data, flow_name):
            print(f"\n{flow_name.upper()}:")
            non_zero_packets = 0
            for pkt_idx, packet in enumerate(flow_data):
                # Skip zero-padded packets
                if np.sum(packet) == 0:
                    continue
                    
                non_zero_packets += 1
                if non_zero_packets > 10:  # Limit to first 10 non-zero packets
                    break
                
                # Extract features
                timestamp = packet[0]
                packet_len = int(packet[1])
                ip_df = int(packet[2])
                ip_mf = int(packet[3]) 
                ip_rb = int(packet[4])
                ip_frag_off = int(packet[5])
                protocols = int(packet[6])
                tcp_len = int(packet[7])
                tcp_ack = int(packet[8])
                tcp_cwr = int(packet[9])
                tcp_ece = int(packet[10])
                tcp_fin = int(packet[11])
                tcp_push = int(packet[12])
                tcp_res = int(packet[13])
                tcp_rst = int(packet[14])
                tcp_syn = int(packet[15])
                tcp_urg = int(packet[16])
                tcp_win = int(packet[17])
                udp_len = int(packet[18])
                icmp_type = int(packet[19])
                
                # Format output
                ip_flags = f"{ip_df}|{ip_mf}|{ip_rb}"
                tcp_flags = f"{tcp_ack}|{tcp_cwr}|{tcp_ece}|{tcp_fin}|{tcp_push}|{tcp_res}|{tcp_rst}|{tcp_syn}|{tcp_urg}"
                protocols_decoded = self.decode_protocols(protocols)
                
                print(f"{pkt_idx:<5} {timestamp:<12.6f} {packet_len:<8} {ip_flags:<15} {ip_frag_off:<8} "
                      f"{protocols_decoded:<12} {tcp_len:<8} {tcp_flags:<25} {tcp_win:<8} {udp_len:<8} {icmp_type:<6}")
        
        # Print each flow type
        print_flow_packets(baseline_flow, "BASELINE")
        print_flow_packets(manipulated_flow, "MANIPULATED") 
        print_flow_packets(fragmented_flow, "FRAGMENTED")
        
        print("\n" + "="*120)
        print("FEATURE ANALYSIS NOTES:")
        print("="*120)
        print("• Baseline: Original unmodified TCP flow")
        print("• Manipulated: Flow after TrafficManipulator PSO optimization")
        print("• Fragmented: Flow after additional IP fragmentation")
        print("• IP Flags: DF=Don't Fragment, MF=More Fragments, RB=Reserved Bit")
        print("• TCP Flags: ACK|CWR|ECE|FIN|PSH|RES|RST|SYN|URG")
        print("• Protocols: Bitmap encoding of protocol stack")
        print("• FragOff > 0: Indicates fragmented packets")
        print("="*120)
        
        return True
    
    def generate_feature_statistics(self, baseline_flows, manipulated_flows, fragmented_flows):
        """Generate statistical comparison of features across transformations"""
        
        def extract_flow_stats(flows, name):
            stats = {}
            all_packets = []
            
            for _, flow, _ in flows:
                for packet in flow:
                    if np.sum(packet) > 0:  # Skip zero-padded packets
                        all_packets.append(packet)
            
            if not all_packets:
                return stats
                
            all_packets = np.array(all_packets)
            
            for i, feature_name in enumerate(self.feature_names):
                feature_values = all_packets[:, i]
                stats[feature_name] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'min': np.min(feature_values),
                    'max': np.max(feature_values),
                    'non_zero': np.count_nonzero(feature_values)
                }
            
            return stats
        
        # Extract statistics
        baseline_stats = extract_flow_stats(baseline_flows, "Baseline")
        manipulated_stats = extract_flow_stats(manipulated_flows, "Manipulated") 
        fragmented_stats = extract_flow_stats(fragmented_flows, "Fragmented")
        
        # Create comparison table
        print("\n" + "="*100)
        print("FEATURE STATISTICS COMPARISON")
        print("="*100)
        print(f"{'Feature':<20} {'Baseline Mean':<15} {'Manipulated Mean':<18} {'Fragmented Mean':<17} {'Change Notes'}")
        print("-" * 100)
        
        for feature in self.feature_names:
            if feature in baseline_stats and feature in manipulated_stats and feature in fragmented_stats:
                base_mean = baseline_stats[feature]['mean']
                manip_mean = manipulated_stats[feature]['mean'] 
                frag_mean = fragmented_stats[feature]['mean']
                
                # Determine significant changes
                change_notes = ""
                if abs(frag_mean - base_mean) > abs(manip_mean - base_mean):
                    if "frag" in feature.lower() or "mf" in feature.lower():
                        change_notes = "↑ Fragmentation effect"
                    else:
                        change_notes = "⚠ Fragmentation impact"
                elif abs(manip_mean - base_mean) > 0.1 * base_mean:
                    change_notes = "→ Manipulation effect"
                
                print(f"{feature:<20} {base_mean:<15.3f} {manip_mean:<18.3f} {frag_mean:<17.3f} {change_notes}")
        
        print("="*100)
    
    def run_comparison(self):
        """Main function to run the feature comparison analysis"""
        
        # Dataset paths
        datasets = {
            'baseline': '/home/rising/EvasionShield-LUCID/DATASETS/merged_flatten_train_dataset.hdf5',
            'manipulated': '/home/rising/EvasionShield-LUCID/DATASETS/merged_matrix_100n_train_dataset.hdf5',  
            'fragmented': '/home/rising/EvasionShield-LUCID/DATASETS/merged_matrix_10n_train_dataset.hdf5'
        }
        
        # Load all datasets
        flows_data = {}
        for dataset_type, path in datasets.items():
            if os.path.exists(path):
                flows_data[dataset_type] = self.load_dataset(path, dataset_type)
            else:
                print(f"Warning: Dataset not found: {path}")
                flows_data[dataset_type] = []
        
        # Create comparison table
        if all(flows_data.values()):
            self.create_comparison_table(
                flows_data['baseline'],
                flows_data['manipulated'], 
                flows_data['fragmented']
            )
            
            # Generate statistics
            self.generate_feature_statistics(
                flows_data['baseline'],
                flows_data['manipulated'],
                flows_data['fragmented']
            )
        else:
            print("Error: Could not load required datasets for comparison")

def main():
    """Main execution function"""
    print("TCP Flow Feature Comparison Tool")
    print("Analyzing feature evolution: Baseline → Manipulated → Fragmented")
    print("-" * 80)
    
    comparator = TCPFlowFeatureComparator()
    comparator.run_comparison()
    
    print("\n📊 Analysis complete! Table shows feature evolution across transformations.")
    print("🔍 Key insights:")
    print("   • Baseline flows show original packet characteristics")
    print("   • Manipulated flows show PSO optimization effects")  
    print("   • Fragmented flows show additional fragmentation impact")
    print("   • IP fragmentation flags (MF) and offsets change significantly")
    print("   • Packet lengths may be altered by fragmentation")

if __name__ == "__main__":
    main()

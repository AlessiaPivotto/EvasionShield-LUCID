#!/usr/bin/env python3
"""
Quick Configuration Test - A faster version with predefined parameter sets
"""

import os
import sys
import json
import time
from datetime import datetime
from manipulator import Manipulator

def create_predefined_configs():
    """Create a set of predefined configurations for common scenarios."""
    
    configs = [
        {
            'config_id': 'conservative',
            'name': 'Conservative Evasion',
            'description': 'Low resource usage, basic evasion',
            'pso_iterations': 3,
            'particle_numbers': 6,
            'local_group_sizes': 3,
            'inertia_weights': 0.7298,
            'cognitive_components': 1.49618,
            'social_components': 1.49618,
            'group_sizes': 100,
            'min_time_extends': 3.0,
            'max_time_extends': 6.0,
            'max_crafted_packets': 1,
            'crafted_packet_probs': 0.01
        },
        {
            'config_id': 'balanced',
            'name': 'Balanced Evasion',
            'description': 'Good balance between evasion and performance',
            'pso_iterations': 10,
            'particle_numbers': 15,
            'local_group_sizes': 3,
            'inertia_weights': 0.7298,
            'cognitive_components': 1.49618,
            'social_components': 1.49618,
            'group_sizes': 100,
            'min_time_extends': 1.0,
            'max_time_extends': 10.0,
            'max_crafted_packets': 2,
            'crafted_packet_probs': 0.05
        },
        {
            'config_id': 'aggressive',
            'name': 'Aggressive Evasion',
            'description': 'Maximum evasion, high resource usage',
            'pso_iterations': 15,
            'particle_numbers': 30,
            'local_group_sizes': 5,
            'inertia_weights': 0.9,
            'cognitive_components': 2.0,
            'social_components': 2.5,
            'group_sizes': 50,
            'min_time_extends': 0.0,
            'max_time_extends': 15.0,
            'max_crafted_packets': 3,
            'crafted_packet_probs': 0.2
        },
        {
            'config_id': 'stealth',
            'name': 'Stealth Mode',
            'description': 'Minimal packet crafting, focus on timing',
            'pso_iterations': 12,
            'particle_numbers': 20,
            'local_group_sizes': 4,
            'inertia_weights': 0.4,
            'cognitive_components': 0.5,
            'social_components': 1.0,
            'group_sizes': 200,
            'min_time_extends': 5.0,
            'max_time_extends': 20.0,
            'max_crafted_packets': 1,
            'crafted_packet_probs': 0.005
        },
        {
            'config_id': 'flooding',
            'name': 'Packet Flooding',
            'description': 'High packet crafting rate',
            'pso_iterations': 8,
            'particle_numbers': 16,
            'local_group_sizes': 4,
            'inertia_weights': 0.8,
            'cognitive_components': 1.8,
            'social_components': 1.8,
            'group_sizes': 75,
            'min_time_extends': 2.0,
            'max_time_extends': 8.0,
            'max_crafted_packets': 5,
            'crafted_packet_probs': 0.15
        },
        {
            'config_id': 'timing_focused',
            'name': 'Timing Manipulation Focus',
            'description': 'Extreme timing manipulation, minimal crafting',
            'pso_iterations': 20,
            'particle_numbers': 25,
            'local_group_sizes': 5,
            'inertia_weights': 0.6,
            'cognitive_components': 1.2,
            'social_components': 2.0,
            'group_sizes': 150,
            'min_time_extends': 0.0,
            'max_time_extends': 25.0,
            'max_crafted_packets': 1,
            'crafted_packet_probs': 0.02
        }
    ]
    
    return configs

def save_config_details(config, config_dir):
    """Save detailed configuration information."""
    config_file = os.path.join(config_dir, "configuration.txt")
    
    with open(config_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CONFIGURATION: {config['name'].upper()}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Config ID: {config['config_id']}\n")
        f.write(f"Description: {config['description']}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PARAMETERS\n")
        f.write("-" * 50 + "\n")
        f.write("PSO Parameters:\n")
        f.write(f"  Max Iterations:      {config['pso_iterations']}\n")
        f.write(f"  Particle Count:      {config['particle_numbers']}\n")
        f.write(f"  Local Group Size:    {config['local_group_sizes']}\n\n")
        
        f.write("Particle Dynamics:\n")
        f.write(f"  Inertia Weight (w):  {config['inertia_weights']}\n")
        f.write(f"  Cognitive (c1):      {config['cognitive_components']}\n")
        f.write(f"  Social (c2):         {config['social_components']}\n\n")
        
        f.write("Manipulation Settings:\n")
        f.write(f"  Packet Group Size:   {config['group_sizes']}\n")
        f.write(f"  Min Time Extension:  {config['min_time_extends']} seconds\n")
        f.write(f"  Max Time Extension:  {config['max_time_extends']} seconds\n")
        f.write(f"  Max Crafted Pkts:    {config['max_crafted_packets']}\n")
        f.write(f"  Crafting Probability:{config['crafted_packet_probs']} ({config['crafted_packet_probs']*100}%)\n\n")
        
        # Calculate derived metrics
        complexity_score = config['pso_iterations'] * config['particle_numbers']
        time_window = config['max_time_extends'] - config['min_time_extends']
        craft_intensity = config['max_crafted_packets'] * config['crafted_packet_probs']
        
        f.write("ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Computational Complexity: {complexity_score} (iter × particles)\n")
        f.write(f"Time Manipulation Range: {time_window:.1f} seconds\n")
        f.write(f"Crafting Intensity:      {craft_intensity:.3f}\n")
        
        if complexity_score >= 400:
            f.write(f"Expected Processing:     Very High\n")
        elif complexity_score >= 200:
            f.write(f"Expected Processing:     High\n")
        elif complexity_score >= 100:
            f.write(f"Expected Processing:     Medium\n")
        else:
            f.write(f"Expected Processing:     Low\n")
            
        if time_window >= 20:
            f.write(f"Timing Evasion:         Extreme\n")
        elif time_window >= 10:
            f.write(f"Timing Evasion:         High\n")
        elif time_window >= 5:
            f.write(f"Timing Evasion:         Medium\n")
        else:
            f.write(f"Timing Evasion:         Conservative\n")

def run_quick_test(pcap_file, mimic_set, normalizer, init_pcap="./data/empty.pcap", output_dir="./quick_test_results"):
    """Run predefined configuration tests."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_dir, f"quick_test_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    configs = create_predefined_configs()
    
    print("=" * 80)
    print("TRAFFICMANIPULATOR QUICK CONFIGURATION TEST")
    print("=" * 80)
    print(f"Input PCAP: {pcap_file}")
    print(f"Results Directory: {results_dir}")
    print(f"Testing {len(configs)} predefined configurations...\n")
    
    results_summary = []
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing: {config['name']}")
        print(f"  Description: {config['description']}")
        
        # Create config directory
        config_dir = os.path.join(results_dir, config['config_id'])
        os.makedirs(config_dir, exist_ok=True)
        
        # Save configuration
        save_config_details(config, config_dir)
        
        start_time = time.time()
        
        try:
            # Initialize manipulator
            m = Manipulator(
                mal_pcap_file=pcap_file,
                mimic_set_file=mimic_set,
                knormer_file=normalizer,
                init_pcap_file=init_pcap
            )
            
            # Set parameters
            m.set_particle_params(
                w=config['inertia_weights'],
                c1=config['cognitive_components'],
                c2=config['social_components']
            )
            
            m.set_pso_params(
                max_iter=config['pso_iterations'],
                particle_num=config['particle_numbers'],
                grp_size=config['local_group_sizes']
            )
            
            m.set_manipulator_params(
                grp_size=config['group_sizes'],
                min_time_extend=config['min_time_extends'],
                max_time_extend=config['max_time_extends'],
                max_cft_pkt=config['max_crafted_packets'],
                max_crafted_pkt_prob=config['crafted_packet_probs']
            )
            
            # Run with limited packets for quick testing
            m.process(start_no=0, limit=200, heuristic=False)  # Quick test with 200 packets
            
            execution_time = time.time() - start_time
            
            # Move generated files
            base_name = os.path.splitext(os.path.basename(pcap_file))[0]
            manipulated_pcap = f"{os.path.splitext(pcap_file)[0]}_manipulated.pcap"
            stats_file = f"{os.path.splitext(pcap_file)[0]}_statistics.pkl"
            
            output_size = 0
            if os.path.exists(manipulated_pcap):
                new_pcap_path = os.path.join(config_dir, f"{base_name}_manipulated.pcap")
                os.rename(manipulated_pcap, new_pcap_path)
                output_size = os.path.getsize(new_pcap_path)
                
            if os.path.exists(stats_file):
                new_stats_path = os.path.join(config_dir, f"{base_name}_statistics.pkl")
                os.rename(stats_file, new_stats_path)
            
            # Save execution results
            results_file = os.path.join(config_dir, "test_results.txt")
            with open(results_file, 'w') as f:
                f.write(f"TEST RESULTS - {config['name']}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Status: SUCCESS\n")
                f.write(f"Execution Time: {execution_time:.2f} seconds\n")
                f.write(f"Output File Size: {output_size} bytes\n")
                f.write(f"Packets Processed: 200 (limited for quick test)\n")
            
            results_summary.append({
                'config_id': config['config_id'],
                'name': config['name'],
                'status': 'SUCCESS',
                'time': execution_time,
                'output_size': output_size
            })
            
            print(f"  ✓ Success ({execution_time:.2f}s)\n")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            results_file = os.path.join(config_dir, "test_results.txt")
            with open(results_file, 'w') as f:
                f.write(f"TEST RESULTS - {config['name']}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Status: FAILED\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Execution Time: {execution_time:.2f} seconds\n")
            
            results_summary.append({
                'config_id': config['config_id'],
                'name': config['name'],
                'status': 'FAILED',
                'time': execution_time,
                'error': str(e)
            })
            
            print(f"  ✗ Failed: {str(e)}\n")
    
    # Create final summary
    summary_file = os.path.join(results_dir, "SUMMARY.txt")
    with open(summary_file, 'w') as f:
        f.write("QUICK TEST SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Run: {timestamp}\n")
        f.write(f"Input File: {pcap_file}\n")
        f.write(f"Results Directory: {results_dir}\n\n")
        
        successful = sum(1 for r in results_summary if r['status'] == 'SUCCESS')
        f.write(f"Total Tests: {len(results_summary)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {len(results_summary) - successful}\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n")
        for result in results_summary:
            f.write(f"{result['config_id']:15} | {result['name']:25} | {result['status']:7} | {result['time']:6.2f}s")
            if 'output_size' in result:
                f.write(f" | {result['output_size']:8} bytes")
            f.write("\n")
    
    print("=" * 80)
    print("QUICK TEST COMPLETED")
    print(f"Successful: {successful}/{len(results_summary)}")
    print(f"Results saved in: {results_dir}")
    print("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test with predefined configurations")
    parser.add_argument('-m', '--mal_pcap', required=True, help="Input malicious PCAP file")
    parser.add_argument('-b', '--mimic_set', required=True, help="Benign features file (.npy)")
    parser.add_argument('-n', '--normalizer', required=True, help="Normalizer file (.pkl)")
    parser.add_argument('-i', '--init_pcap', default='./data/empty.pcap', help="Init PCAP file")
    parser.add_argument('-o', '--output_dir', default='./quick_test_results', help="Output directory")
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path, name in [(args.mal_pcap, "PCAP"), (args.mimic_set, "Mimic set"), (args.normalizer, "Normalizer")]:
        if not os.path.exists(file_path):
            print(f"Error: {name} file not found: {file_path}")
            sys.exit(1)
    
    run_quick_test(args.mal_pcap, args.mimic_set, args.normalizer, args.init_pcap, args.output_dir)

if __name__ == "__main__":
    main()

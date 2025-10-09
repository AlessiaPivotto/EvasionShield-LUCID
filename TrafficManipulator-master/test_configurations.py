#!/usr/bin/env python3
"""
Configuration Testing Script for TrafficManipulator
This script tests multiple parameter combinations to find optimal evasion configurations.
"""

import os
import sys
import json
import time
import itertools
import numpy as np
from datetime import datetime
from manipulator import Manipulator

class ConfigurationTester:
    def __init__(self, base_output_dir="./configuration_tests"):
        self.base_output_dir = base_output_dir
        self.test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(base_output_dir, f"test_run_{self.test_timestamp}")
        
        # Create base directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configuration parameter ranges
        self.param_ranges = {
            # PSO Parameters
            'pso_iterations': [3, 5, 10, 15],
            'particle_numbers': [6, 10, 20, 30],
            'local_group_sizes': [3, 5],
            
            # Particle Parameters  
            'inertia_weights': [0.4, 0.7298, 0.9],
            'cognitive_components': [0.5, 1.49618, 2.0],
            'social_components': [1.0, 1.49618, 2.5],
            
            # Manipulator Parameters
            'group_sizes': [50, 100, 200],
            'min_time_extends': [0.0, 1.0, 3.0],
            'max_time_extends': [5.0, 6.0, 10.0, 15.0],
            'max_crafted_packets': [1, 2, 3],
            'crafted_packet_probs': [0.01, 0.05, 0.1, 0.2]
        }
        
    def generate_configurations(self, max_configs=50):
        """Generate different parameter combinations."""
        print(f"Generating configurations (max: {max_configs})...")
        
        # Create all possible combinations
        param_names = list(self.param_ranges.keys())
        param_values = [self.param_ranges[name] for name in param_names]
        
        all_combinations = list(itertools.product(*param_values))
        
        # If too many combinations, sample randomly
        if len(all_combinations) > max_configs:
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(len(all_combinations), max_configs, replace=False)
            selected_combinations = [all_combinations[i] for i in selected_indices]
        else:
            selected_combinations = all_combinations
            
        # Convert to configuration dictionaries
        configurations = []
        for i, combo in enumerate(selected_combinations):
            config = {param_names[j]: combo[j] for j in range(len(param_names))}
            config['config_id'] = f"config_{i+1:03d}"
            configurations.append(config)
            
        print(f"Generated {len(configurations)} configurations")
        return configurations
    
    def save_configuration(self, config, config_dir):
        """Save configuration to file."""
        config_file = os.path.join(config_dir, "configuration.txt")
        
        with open(config_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TRAFFICMANIPULATOR CONFIGURATION - {config['config_id'].upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("PSO (Particle Swarm Optimization) Parameters:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Max Iterations:      {config['pso_iterations']}\n")
            f.write(f"  Particle Number:     {config['particle_numbers']}\n")
            f.write(f"  Local Group Size:    {config['local_group_sizes']}\n\n")
            
            f.write("Particle Parameters:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Inertia Weight (w):  {config['inertia_weights']}\n")
            f.write(f"  Cognitive (c1):      {config['cognitive_components']}\n")
            f.write(f"  Social (c2):         {config['social_components']}\n\n")
            
            f.write("Manipulator Parameters:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Group Size:          {config['group_sizes']}\n")
            f.write(f"  Min Time Extend:     {config['min_time_extends']} seconds\n")
            f.write(f"  Max Time Extend:     {config['max_time_extends']} seconds\n")
            f.write(f"  Max Crafted Packets: {config['max_crafted_packets']}\n")
            f.write(f"  Crafted Packet Prob: {config['crafted_packet_probs']} ({config['crafted_packet_probs']*100}%)\n\n")
            
            f.write("Configuration Summary:\n")
            f.write("-" * 50 + "\n")
            evasion_level = self.estimate_evasion_level(config)
            f.write(f"  Estimated Evasion Level: {evasion_level}\n")
            f.write(f"  Processing Intensity:    {self.estimate_processing_intensity(config)}\n")
            f.write(f"  Time Manipulation:       {self.estimate_time_manipulation(config)}\n")
            f.write(f"  Packet Crafting:         {self.estimate_packet_crafting(config)}\n\n")
            
            # Save as JSON for programmatic access
            json_file = os.path.join(config_dir, "configuration.json")
            with open(json_file, 'w') as jf:
                json.dump(config, jf, indent=2)
    
    def estimate_evasion_level(self, config):
        """Estimate evasion effectiveness based on parameters."""
        score = 0
        
        # Higher iterations = better optimization
        if config['pso_iterations'] >= 15: score += 3
        elif config['pso_iterations'] >= 10: score += 2
        elif config['pso_iterations'] >= 5: score += 1
        
        # More particles = better search
        if config['particle_numbers'] >= 30: score += 2
        elif config['particle_numbers'] >= 20: score += 1
        
        # Higher time extension = more evasion
        if config['max_time_extends'] >= 15: score += 3
        elif config['max_time_extends'] >= 10: score += 2
        elif config['max_time_extends'] >= 6: score += 1
        
        # More crafted packets = more evasion
        if config['max_crafted_packets'] >= 3: score += 2
        elif config['max_crafted_packets'] >= 2: score += 1
        
        # Higher crafting probability = more evasion
        if config['crafted_packet_probs'] >= 0.2: score += 2
        elif config['crafted_packet_probs'] >= 0.1: score += 1
        
        if score >= 10: return "Very High"
        elif score >= 7: return "High"
        elif score >= 4: return "Medium"
        else: return "Low"
    
    def estimate_processing_intensity(self, config):
        """Estimate computational intensity."""
        intensity = config['pso_iterations'] * config['particle_numbers']
        if intensity >= 300: return "Very High"
        elif intensity >= 150: return "High"  
        elif intensity >= 50: return "Medium"
        else: return "Low"
    
    def estimate_time_manipulation(self, config):
        """Estimate time manipulation level."""
        time_range = config['max_time_extends'] - config['min_time_extends']
        if time_range >= 15: return "Aggressive"
        elif time_range >= 10: return "High"
        elif time_range >= 5: return "Medium"
        else: return "Conservative"
    
    def estimate_packet_crafting(self, config):
        """Estimate packet crafting level."""
        crafting_score = config['max_crafted_packets'] * config['crafted_packet_probs']
        if crafting_score >= 0.6: return "Aggressive"
        elif crafting_score >= 0.2: return "High"
        elif crafting_score >= 0.05: return "Medium"
        else: return "Conservative"
    
    def run_configuration_test(self, config, pcap_file, mimic_set, normalizer, init_pcap):
        """Run TrafficManipulator with a specific configuration."""
        print(f"\nTesting {config['config_id']}...")
        print(f"  Evasion Level: {self.estimate_evasion_level(config)}")
        print(f"  Processing: {self.estimate_processing_intensity(config)}")
        
        # Create configuration directory
        config_dir = os.path.join(self.results_dir, config['config_id'])
        os.makedirs(config_dir, exist_ok=True)
        
        # Save configuration
        self.save_configuration(config, config_dir)
        
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
            
            # Run processing with limited packets for testing (change limit for full run)
            m.process(start_no=0, limit=500, heuristic=False)  # Limit to 500 packets for testing
            
            # Move generated files to configuration directory
            base_name = os.path.splitext(os.path.basename(pcap_file))[0]
            manipulated_pcap = f"{os.path.splitext(pcap_file)[0]}_manipulated.pcap"
            stats_file = f"{os.path.splitext(pcap_file)[0]}_statistics.pkl"
            
            if os.path.exists(manipulated_pcap):
                new_pcap_path = os.path.join(config_dir, f"{base_name}_manipulated.pcap")
                os.rename(manipulated_pcap, new_pcap_path)
                
            if os.path.exists(stats_file):
                new_stats_path = os.path.join(config_dir, f"{base_name}_statistics.pkl")
                os.rename(stats_file, new_stats_path)
            
            execution_time = time.time() - start_time
            
            # Save execution results
            results_file = os.path.join(config_dir, "execution_results.txt")
            with open(results_file, 'w') as f:
                f.write(f"Execution Results - {config['config_id']}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Execution Time: {execution_time:.2f} seconds\n")
                f.write(f"Status: SUCCESS\n")
                f.write(f"Input File: {pcap_file}\n")
                if os.path.exists(new_pcap_path):
                    f.write(f"Output File: {new_pcap_path}\n")
                    f.write(f"Output Size: {os.path.getsize(new_pcap_path)} bytes\n")
            
            print(f"  ✓ Completed in {execution_time:.2f}s")
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"  ✗ Failed: {str(e)}")
            
            # Save error results
            results_file = os.path.join(config_dir, "execution_results.txt")
            with open(results_file, 'w') as f:
                f.write(f"Execution Results - {config['config_id']}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Execution Time: {execution_time:.2f} seconds\n")
                f.write(f"Status: FAILED\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Input File: {pcap_file}\n")
            
            return False
    
    def run_test_suite(self, pcap_file, mimic_set, normalizer, init_pcap="./data/empty.pcap", max_configs=20):
        """Run the complete test suite."""
        print("=" * 80)
        print("TRAFFICMANIPULATOR CONFIGURATION TESTING SUITE")
        print("=" * 80)
        print(f"Test Run ID: {self.test_timestamp}")
        print(f"Results Directory: {self.results_dir}")
        print(f"Input PCAP: {pcap_file}")
        print(f"Max Configurations: {max_configs}")
        
        # Generate configurations
        configurations = self.generate_configurations(max_configs)
        
        # Create summary file
        summary_file = os.path.join(self.results_dir, "test_summary.txt")
        
        successful_tests = 0
        failed_tests = 0
        
        with open(summary_file, 'w') as sf:
            sf.write("TRAFFICMANIPULATOR CONFIGURATION TEST SUMMARY\n")
            sf.write("=" * 80 + "\n")
            sf.write(f"Test Run: {self.test_timestamp}\n")
            sf.write(f"Input File: {pcap_file}\n")
            sf.write(f"Total Configurations: {len(configurations)}\n\n")
            
            # Run each configuration
            for i, config in enumerate(configurations, 1):
                print(f"\n[{i}/{len(configurations)}] Running {config['config_id']}...")
                
                success = self.run_configuration_test(config, pcap_file, mimic_set, normalizer, init_pcap)
                
                if success:
                    successful_tests += 1
                    status = "SUCCESS"
                else:
                    failed_tests += 1
                    status = "FAILED"
                
                # Write to summary
                sf.write(f"{config['config_id']}: {status}\n")
                sf.write(f"  Evasion Level: {self.estimate_evasion_level(config)}\n")
                sf.write(f"  Processing: {self.estimate_processing_intensity(config)}\n")
                sf.write(f"  Parameters: iter={config['pso_iterations']}, ")
                sf.write(f"particles={config['particle_numbers']}, ")
                sf.write(f"time_ext={config['max_time_extends']}, ")
                sf.write(f"craft_prob={config['crafted_packet_probs']}\n\n")
                sf.flush()
        
        # Final summary
        print(f"\n{'='*80}")
        print("TEST SUITE COMPLETED")
        print(f"{'='*80}")
        print(f"Total Tests: {len(configurations)}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/len(configurations)*100):.1f}%")
        print(f"Results saved in: {self.results_dir}")
        
        # Update summary with final results
        with open(summary_file, 'a') as sf:
            sf.write("FINAL RESULTS\n")
            sf.write("-" * 40 + "\n")
            sf.write(f"Total Tests: {len(configurations)}\n")
            sf.write(f"Successful: {successful_tests}\n")
            sf.write(f"Failed: {failed_tests}\n")
            sf.write(f"Success Rate: {(successful_tests/len(configurations)*100):.1f}%\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TrafficManipulator with different configurations")
    parser.add_argument('-m', '--mal_pcap', required=True, help="Input malicious PCAP file")
    parser.add_argument('-b', '--mimic_set', required=True, help="Benign features file (.npy)")
    parser.add_argument('-n', '--normalizer', required=True, help="Normalizer file (.pkl)")
    parser.add_argument('-i', '--init_pcap', default='./data/empty.pcap', help="Init PCAP file")
    parser.add_argument('-o', '--output_dir', default='./configuration_tests', help="Output directory")
    parser.add_argument('-c', '--max_configs', type=int, default=20, help="Maximum configurations to test")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.mal_pcap):
        print(f"Error: PCAP file not found: {args.mal_pcap}")
        sys.exit(1)
    
    if not os.path.exists(args.mimic_set):
        print(f"Error: Mimic set file not found: {args.mimic_set}")
        sys.exit(1)
        
    if not os.path.exists(args.normalizer):
        print(f"Error: Normalizer file not found: {args.normalizer}")
        sys.exit(1)
    
    # Create tester and run
    tester = ConfigurationTester(args.output_dir)
    tester.run_test_suite(
        pcap_file=args.mal_pcap,
        mimic_set=args.mimic_set,
        normalizer=args.normalizer,
        init_pcap=args.init_pcap,
        max_configs=args.max_configs
    )


if __name__ == "__main__":
    main()

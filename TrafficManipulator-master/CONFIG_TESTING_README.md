# TrafficManipulator Configuration Testing Suite

This directory contains comprehensive testing tools to evaluate different TrafficManipulator configurations and find optimal evasion parameters.

## Files Overview

### Testing Scripts
- **`test_configurations.py`** - Comprehensive testing with randomly generated parameter combinations
- **`quick_test_configs.py`** - Fast testing with 6 predefined configuration scenarios  
- **`run_config_tests.sh`** - Bash wrapper for batch testing multiple PCAP files
- **`main.py`** - Original TrafficManipulator with enhanced default parameters

### Configuration Documentation
- **`CONFIG_TESTING_README.md`** - This file

## Quick Start

### 1. Quick Test (Recommended for initial evaluation)
Test 6 predefined configurations optimized for different scenarios:

```bash
# Test with a specific PCAP file
python3 quick_test_configs.py -m your_file.pcap -b example/mimic_set.npy -n example/normalizer.pkl

# Or use the batch wrapper
./run_config_tests.sh --quick your_file.pcap
```

### 2. Full Configuration Test
Generate and test up to 50 random parameter combinations:

```bash
# Test 20 random configurations  
python3 test_configurations.py -m your_file.pcap -b example/mimic_set.npy -n example/normalizer.pkl -c 20

# Test 50 configurations
python3 test_configurations.py -m your_file.pcap -b example/mimic_set.npy -n example/normalizer.pkl -c 50
```

### 3. Batch Testing
Automatically test multiple PCAP files:

```bash
# Auto-detect and test all PCAP files
./run_config_tests.sh --auto

# Test specific file with full suite
./run_config_tests.sh --full your_file.pcap
```

## Predefined Configurations (Quick Test)

The quick test includes these optimized scenarios:

| Configuration | Description | Use Case |
|--------------|-------------|----------|
| **Conservative** | Low resource, basic evasion | Production environments |
| **Balanced** | Good evasion/performance balance | General purpose |
| **Aggressive** | Maximum evasion, high resource | High-security targets |
| **Stealth** | Minimal crafting, timing focus | Avoiding detection |
| **Flooding** | High packet crafting rate | Overwhelming defenses |
| **Timing Focused** | Extreme timing manipulation | IDS timeout exploitation |

## Parameter Ranges (Full Test)

The comprehensive test explores these parameter ranges:

### PSO Parameters
- **Iterations**: 3, 5, 10, 15
- **Particles**: 6, 10, 20, 30  
- **Group Size**: 3, 5

### Particle Dynamics
- **Inertia Weight (w)**: 0.4, 0.7298, 0.9
- **Cognitive (c1)**: 0.5, 1.49618, 2.0
- **Social (c2)**: 1.0, 1.49618, 2.5

### Manipulation Parameters
- **Group Size**: 50, 100, 200 packets
- **Min Time Extension**: 0.0, 1.0, 3.0 seconds
- **Max Time Extension**: 5.0, 6.0, 10.0, 15.0 seconds
- **Max Crafted Packets**: 1, 2, 3
- **Crafting Probability**: 0.01, 0.05, 0.1, 0.2 (1%-20%)

## Output Structure

Each test creates a structured output directory:

```
configuration_tests_YYYYMMDD_HHMMSS/
├── test_summary.txt                    # Overall results summary
├── config_001/                         # Individual configuration results
│   ├── configuration.txt               # Human-readable config
│   ├── configuration.json              # Machine-readable config  
│   ├── execution_results.txt           # Test execution details
│   ├── your_file_manipulated.pcap      # Output manipulated PCAP
│   └── your_file_statistics.pkl        # Processing statistics
├── config_002/
│   └── ... (same structure)
└── config_N/
```

## Configuration File Format

Each configuration directory contains detailed information:

```
================================================================================================
CONFIGURATION: AGGRESSIVE EVASION
================================================================================================
Config ID: aggressive
Description: Maximum evasion, high resource usage
Generated: 2025-10-06 14:30:15

PARAMETERS
--------------------------------------------------
PSO Parameters:
  Max Iterations:      15
  Particle Count:      30
  Local Group Size:    5

Particle Dynamics:
  Inertia Weight (w):  0.9
  Cognitive (c1):      2.0
  Social (c2):         2.5

Manipulation Settings:
  Packet Group Size:   50
  Min Time Extension:  0.0 seconds
  Max Time Extension:  15.0 seconds
  Max Crafted Pkts:    3
  Crafting Probability:0.2 (20.0%)

ANALYSIS
--------------------------------------------------
Computational Complexity: 450 (iter × particles)
Time Manipulation Range: 15.0 seconds
Crafting Intensity:      0.600
Expected Processing:     Very High
Timing Evasion:         Extreme
```

## Command Line Options

### test_configurations.py
```
-m, --mal_pcap      Input malicious PCAP file (required)
-b, --mimic_set     Benign features file .npy (required)  
-n, --normalizer    Normalizer file .pkl (required)
-i, --init_pcap     Init PCAP file (default: ./data/empty.pcap)
-o, --output_dir    Output directory (default: ./configuration_tests)
-c, --max_configs   Maximum configurations to test (default: 20)
```

### quick_test_configs.py
```
-m, --mal_pcap      Input malicious PCAP file (required)
-b, --mimic_set     Benign features file .npy (required)
-n, --normalizer    Normalizer file .pkl (required)  
-i, --init_pcap     Init PCAP file (default: ./data/empty.pcap)
-o, --output_dir    Output directory (default: ./quick_test_results)
```

### run_config_tests.sh
```
./run_config_tests.sh <pcap_file>          # Test specific file
./run_config_tests.sh --auto               # Auto-detect PCAP files
./run_config_tests.sh --quick <pcap_file>   # Quick test (6 configs)
./run_config_tests.sh --full <pcap_file>    # Full test (20+ configs)
```

## Analyzing Results

### 1. Success Rate Analysis
Check `test_summary.txt` for overall success rates:
- Configurations that failed to complete
- Execution times for different parameter sets
- Resource usage patterns

### 2. Evasion Effectiveness
Compare different approaches:
- **Conservative**: Low resource, minimal evasion
- **Aggressive**: Maximum evasion, high computational cost  
- **Balanced**: Optimal evasion/performance ratio

### 3. Performance Metrics
- **Execution Time**: How long each configuration takes
- **Output File Size**: Indicates manipulation intensity
- **Success Rate**: Reliability of parameter combinations

### 4. Optimal Configuration Selection
Consider these factors:
- **Target Environment**: Production vs. testing
- **Available Resources**: CPU, memory, time constraints
- **Evasion Requirements**: Basic vs. advanced IDS systems
- **Stealth Requirements**: Minimal vs. aggressive manipulation

## Best Practices

### For Production Use
1. Start with **Conservative** or **Balanced** configurations
2. Test with representative traffic samples first
3. Monitor resource usage during execution
4. Validate output PCAP files before deployment

### For Research/Testing
1. Use **Aggressive** or **Timing Focused** for maximum evasion
2. Run full configuration tests to explore parameter space
3. Compare results across different attack types
4. Document successful configurations for reuse

### Resource Management
1. Limit packet count for initial tests (`limit=200-500`)
2. Monitor memory usage with large PCAP files
3. Use batch processing for multiple files
4. Clean up intermediate files after testing

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce `group_size` or `particle_num`
2. **Timeout errors**: Reduce `pso_iterations` or `max_time_extend`
3. **Missing files**: Check file paths in error messages
4. **Permission errors**: Ensure write access to output directory

### Performance Optimization
1. **Fast testing**: Use `limit=100-500` packets initially
2. **Resource constraints**: Reduce `particle_num` and `pso_iterations`  
3. **Large files**: Split PCAP files into smaller chunks
4. **Parallel processing**: Run multiple configurations on different cores

## Configuration Recommendations

### By Target Type
- **Web Application IDS**: Use `stealth` or `timing_focused`
- **Network IDS**: Use `aggressive` or `flooding`  
- **Signature-based IDS**: Use `balanced` with high crafting probability
- **Behavioral IDS**: Use `timing_focused` with minimal crafting

### By Evasion Goal
- **Bypass rate limiting**: High `max_time_extend`
- **Avoid signature detection**: High `crafting_probability`
- **Resource exhaustion**: `flooding` configuration
- **Stealth operation**: `stealth` configuration with low crafting

## Example Workflows

### 1. Initial Assessment
```bash
# Quick test to understand baseline performance
./run_config_tests.sh --quick sample.pcap

# Analyze results
ls -la quick_test_results_*/*/
cat quick_test_results_*/*/SUMMARY.txt
```

### 2. Optimization Phase  
```bash
# Full parameter exploration
python3 test_configurations.py -m sample.pcap -b example/mimic_set.npy -n example/normalizer.pkl -c 30

# Find best performing configurations
grep -r "SUCCESS" configuration_tests_*/config_*/execution_results.txt
```

### 3. Production Deployment
```bash
# Test selected configuration with full dataset
python3 main.py -m full_dataset.pcap -b example/mimic_set.npy -n example/normalizer.pkl

# Verify output quality
tcpdump -r full_dataset_manipulated.pcap -c 100
```

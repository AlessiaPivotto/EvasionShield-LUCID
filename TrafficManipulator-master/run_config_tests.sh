#!/bin/bash
"""
Batch Testing Script for TrafficManipulator Configurations
This script runs configuration tests on multiple PCAP files
"""

# Configuration
MIMIC_SET="example/mimic_set.npy"
NORMALIZER="example/normalizer.pkl"  
INIT_PCAP="example/init.pcap"
TEST_TYPE="quick"  # "quick" or "full"
MAX_CONFIGS=20

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required files exist
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_files=0
    
    if [[ ! -f "$MIMIC_SET" ]]; then
        print_error "Mimic set file not found: $MIMIC_SET"
        missing_files=$((missing_files + 1))
    fi
    
    if [[ ! -f "$NORMALIZER" ]]; then
        print_error "Normalizer file not found: $NORMALIZER"
        missing_files=$((missing_files + 1))
    fi
    
    if [[ ! -f "$INIT_PCAP" ]]; then
        print_warning "Init PCAP file not found: $INIT_PCAP (will use default)"
        INIT_PCAP="./data/empty.pcap"
    fi
    
    if [[ $missing_files -gt 0 ]]; then
        print_error "Missing $missing_files required files. Exiting."
        exit 1
    fi
    
    print_status "All dependencies found ✓"
}

# Find PCAP files to test
find_pcap_files() {
    print_status "Searching for PCAP files..."
    
    # Look for PCAP files in common locations
    local pcap_files=()
    
    # Check current directory
    while IFS= read -r -d '' file; do
        pcap_files+=("$file")
    done < <(find . -maxdepth 2 -name "*.pcap" -type f -print0 2>/dev/null)
    
    # Check example directory
    while IFS= read -r -d '' file; do
        pcap_files+=("$file")
    done < <(find example/ -name "*.pcap" -type f -print0 2>/dev/null)
    
    # Check data directory  
    while IFS= read -r -d '' file; do
        pcap_files+=("$file")
    done < <(find data/ -name "*.pcap" -type f -print0 2>/dev/null)
    
    if [[ ${#pcap_files[@]} -eq 0 ]]; then
        print_error "No PCAP files found. Please specify PCAP file manually."
        echo "Usage: $0 <pcap_file>"
        echo "   or: $0 --auto  (to search automatically)"
        exit 1
    fi
    
    echo "${pcap_files[@]}"
}

# Run configuration test on a single PCAP file
test_single_pcap() {
    local pcap_file="$1"
    local pcap_name=$(basename "$pcap_file" .pcap)
    
    print_status "Testing configuration with: $pcap_file"
    
    # Create unique output directory
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_dir="./config_tests_${pcap_name}_${timestamp}"
    
    if [[ "$TEST_TYPE" == "quick" ]]; then
        print_status "Running quick test (6 predefined configurations)..."
        python3 quick_test_configs.py \
            -m "$pcap_file" \
            -b "$MIMIC_SET" \
            -n "$NORMALIZER" \
            -i "$INIT_PCAP" \
            -o "$output_dir"
    else
        print_status "Running full test (up to $MAX_CONFIGS configurations)..."
        python3 test_configurations.py \
            -m "$pcap_file" \
            -b "$MIMIC_SET" \
            -n "$NORMALIZER" \
            -i "$INIT_PCAP" \
            -o "$output_dir" \
            -c "$MAX_CONFIGS"
    fi
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        print_status "✓ Configuration test completed successfully"
        print_status "Results saved in: $output_dir"
        
        # Show summary if available
        if [[ -f "$output_dir"/*/SUMMARY.txt ]]; then
            echo ""
            print_status "Test Summary:"
            echo -e "${BLUE}$(cat "$output_dir"/*/SUMMARY.txt | head -10)${NC}"
        fi
        
        return 0
    else
        print_error "✗ Configuration test failed with exit code $exit_code"
        return 1
    fi
}

# Main function
main() {
    print_header "TRAFFICMANIPULATOR CONFIGURATION BATCH TESTER"
    
    # Parse command line arguments
    if [[ $# -eq 0 ]]; then
        print_error "No arguments provided."
        echo "Usage:"
        echo "  $0 <pcap_file>              # Test specific PCAP file"
        echo "  $0 --auto                   # Auto-detect PCAP files"
        echo "  $0 --quick <pcap_file>      # Quick test (6 configs)"
        echo "  $0 --full <pcap_file>       # Full test (20+ configs)"
        exit 1
    fi
    
    # Process arguments
    if [[ "$1" == "--auto" ]]; then
        check_dependencies
        local pcap_files=($(find_pcap_files))
        
        print_status "Found ${#pcap_files[@]} PCAP files:"
        for pcap in "${pcap_files[@]}"; do
            echo "  - $pcap"
        done
        
        echo ""
        read -p "Proceed with testing all files? (y/N): " -n 1 -r
        echo
        
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Cancelled by user."
            exit 0
        fi
        
        # Test each PCAP file
        local total_tests=${#pcap_files[@]}
        local successful_tests=0
        
        for i in "${!pcap_files[@]}"; do
            local pcap_file="${pcap_files[$i]}"
            local current=$((i + 1))
            
            print_header "Testing PCAP $current/$total_tests: $(basename "$pcap_file")"
            
            if test_single_pcap "$pcap_file"; then
                successful_tests=$((successful_tests + 1))
            fi
            
            echo ""
        done
        
        print_header "BATCH TESTING COMPLETED"
        print_status "Total files tested: $total_tests"
        print_status "Successful tests: $successful_tests"
        print_status "Failed tests: $((total_tests - successful_tests))"
        
    elif [[ "$1" == "--quick" ]]; then
        TEST_TYPE="quick"
        shift
        if [[ $# -eq 0 ]]; then
            print_error "PCAP file required for --quick option"
            exit 1
        fi
        check_dependencies
        test_single_pcap "$1"
        
    elif [[ "$1" == "--full" ]]; then
        TEST_TYPE="full"
        shift
        if [[ $# -eq 0 ]]; then
            print_error "PCAP file required for --full option"
            exit 1
        fi
        check_dependencies
        test_single_pcap "$1"
        
    elif [[ -f "$1" ]]; then
        check_dependencies
        test_single_pcap "$1"
        
    else
        print_error "File not found: $1"
        exit 1
    fi
}

# Run main function
main "$@"

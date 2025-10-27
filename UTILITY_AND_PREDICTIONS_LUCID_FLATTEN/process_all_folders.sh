#!/bin/bash
# Script to process all folders in MANIPULATED_FLATTEN_42 directory
# Executes lucid_dataset_parser.py with both dataset processing and preprocessing commands

# Base directory containing all folders to process
BASE_DIR="./TrafficManipulator-master/DATASETS/MANIPULATED_FLATTEN_42"

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist"
    exit 1
fi

# Get all subdirectories and process them
echo "Processing all folders in $BASE_DIR"
echo "=================================="

# Initialize counters
successful_count=0
failed_count=0
declare -a successful_folders
declare -a failed_folders

# Process each folder
for folder_path in "$BASE_DIR"/*; do
    if [ -d "$folder_path" ]; then
        folder_name=$(basename "$folder_path")
        echo ""
        echo "🚀 Processing folder: $folder_name"
        echo "================================================="
        
        # Command 1: Dataset processing
        echo "Running dataset processing..."
        if python3 flatten_lucid/lucid_dataset_parser.py \
            --dataset_type DOS2019 \
            --dataset_folder "$folder_path" \
            --packets_per_flow 100 \
            --dataset_id DOS2019 \
            --traffic_type all \
            --time_window 10; then
            echo "✅ Dataset processing completed for $folder_name"
            cmd1_success=true
        else
            echo "❌ Dataset processing failed for $folder_name"
            cmd1_success=false
        fi
        
        # Command 2: Preprocessing with flatten
        echo "Running preprocessing with flatten..."
        if python3 flatten_lucid/lucid_dataset_parser.py \
            --preprocess_folder "$folder_path" \
            --flatten; then
            echo "✅ Preprocessing completed for $folder_name"
            cmd2_success=true
        else
            echo "❌ Preprocessing failed for $folder_name"
            cmd2_success=false
        fi
        
        # Check overall success for this folder
        if $cmd1_success && $cmd2_success; then
            echo "✅ Successfully processed folder: $folder_name"
            successful_folders+=("$folder_name")
            ((successful_count++))
        else
            echo "❌ Failed to process folder: $folder_name"
            failed_folders+=("$folder_name")
            ((failed_count++))
        fi
    fi
done

# Summary
echo ""
echo "==============================================="
echo "PROCESSING SUMMARY"
echo "==============================================="
echo "Total folders processed: $((successful_count + failed_count))"
echo "Successful: $successful_count"
echo "Failed: $failed_count"

if [ ${#successful_folders[@]} -gt 0 ]; then
    echo ""
    echo "✅ Successfully processed folders:"
    for folder in "${successful_folders[@]}"; do
        echo "  - $folder"
    done
fi

if [ ${#failed_folders[@]} -gt 0 ]; then
    echo ""
    echo "❌ Failed folders:"
    for folder in "${failed_folders[@]}"; do
        echo "  - $folder"
    done
    exit 1
else
    echo ""
    echo "🎉 All folders processed successfully!"
fi

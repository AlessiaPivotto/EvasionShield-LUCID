#!/usr/bin/env python3
"""
Script to process all folders in MANIPULATED_FLATTEN_42 directory
Executes lucid_dataset_parser.py with both dataset processing and preprocessing commands
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, folder_name):
    """Execute a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder_name}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("✅ Command executed successfully")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    # Base directory containing all folders to process
    base_dir = "./TrafficManipulator-master/DATASETS/FRAGMENTED_42_150"
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)
    
    # Get all subdirectories
    folders = [f for f in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, f))]
    folders.sort()  # Sort for consistent processing order
    
    if not folders:
        print(f"No folders found in {base_dir}")
        sys.exit(1)
    
    print(f"Found {len(folders)} folders to process:")
    for folder in folders:
        print(f"  - {folder}")
    
    # Process each folder
    successful_folders = []
    failed_folders = []
    
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        
        print(f"\n🚀 Starting processing for folder: {folder}")
        
        # Command 1: Dataset processing
        cmd1 = [
            "python3", "flatten_lucid/lucid_dataset_parser.py",
            "--dataset_type", "DOS2019",
            "--dataset_folder", folder_path,
            "--packets_per_flow", "100",
            "--dataset_id", "DOS2019",
            "--traffic_type", "all",
            "--time_window", "10"
        ]
        
        # Command 2: Preprocessing with flatten
        cmd2 = [
            "python3", "flatten_lucid/lucid_dataset_parser.py",
            "--preprocess_folder", folder_path,
            "--flatten",
            "--dont_normalize"
        ]
        
        # Execute both commands
        success1 = run_command(cmd1, f"{folder} (Dataset Processing)")
        success2 = run_command(cmd2, f"{folder} (Preprocessing)")
        
        if success1 and success2:
            successful_folders.append(folder)
            print(f"✅ Successfully processed folder: {folder}")
        else:
            failed_folders.append(folder)
            print(f"❌ Failed to process folder: {folder}")
    
    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total folders: {len(folders)}")
    print(f"Successful: {len(successful_folders)}")
    print(f"Failed: {len(failed_folders)}")
    
    if successful_folders:
        print(f"\n✅ Successfully processed folders:")
        for folder in successful_folders:
            print(f"  - {folder}")
    
    if failed_folders:
        print(f"\n❌ Failed folders:")
        for folder in failed_folders:
            print(f"  - {folder}")
        sys.exit(1)
    else:
        print(f"\n🎉 All folders processed successfully!")

if __name__ == "__main__":
    main()

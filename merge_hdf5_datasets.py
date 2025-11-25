#!/usr/bin/env python3

"""
Merge HDF5 files (train, test, val) into single datasets for fragmentation analysis.

This allows us to have all available samples for testing fragmentation effectiveness,
rather than being limited by train/test/val splits.
"""

import h5py
import numpy as np
import os
from pathlib import Path
import sys

def merge_hdf5_files(base_dir, attack_type, output_suffix="_merged"):
    """Merge train, test, val HDF5 files for a specific attack type"""
    
    attack_dir = Path(base_dir) / attack_type
    
    if not attack_dir.exists():
        print(f"❌ Directory not found: {attack_dir}")
        return False
    
    # Look for HDF5 files
    patterns = ["*train*.hdf5", "*test*.hdf5", "*val*.hdf5"]
    all_files = []
    
    for pattern in patterns:
        files = list(attack_dir.glob(pattern))
        all_files.extend(files)
    
    if not all_files:
        print(f"❌ No HDF5 files found in {attack_dir}")
        return False
    
    print(f"📂 Found {len(all_files)} HDF5 files in {attack_type}:")
    for f in all_files:
        print(f"   - {f.name}")
    
    # Load and merge data
    all_features = []
    all_labels = []
    all_flow_ids = []
    total_samples = 0
    
    for hdf5_file in all_files:
        try:
            with h5py.File(hdf5_file, 'r') as f:
                # Check what keys exist
                keys = list(f.keys())
                print(f"   📊 {hdf5_file.name}: {keys}")
                
                # Load features and labels
                features = f['set_x'][:]
                labels = f['set_y'][:]
                
                print(f"      Loaded {len(features)} samples")
                
                all_features.append(features)
                all_labels.append(labels)
                
                # Load flow_ids if available
                if 'flow_ids' in keys:
                    flow_ids = f['flow_ids'][:]
                    all_flow_ids.append(flow_ids)
                
                total_samples += len(features)
                
        except Exception as e:
            print(f"   ❌ Error loading {hdf5_file.name}: {e}")
            continue
    
    if not all_features:
        print(f"❌ No valid data loaded from {attack_type}")
        return False
    
    # Concatenate all data
    merged_features = np.vstack(all_features)
    merged_labels = np.hstack(all_labels)
    
    print(f"✅ Merged data: {merged_features.shape[0]} samples, {merged_features.shape[1]} features")
    
    # Check label distribution
    unique_labels, counts = np.unique(merged_labels, return_counts=True)
    print(f"   Label distribution:")
    for label, count in zip(unique_labels, counts):
        class_name = "benign" if label == 0 else "attack"
        percentage = (count/len(merged_labels))*100
        print(f"   - Class {label} ({class_name}): {count} samples ({percentage:.1f}%)")
    
    # Create output file
    base_filename = all_files[0].name.replace('-train', '').replace('-test', '').replace('-val', '')
    if base_filename.endswith('.hdf5'):
        base_filename = base_filename[:-5]
    
    output_file = attack_dir / f"{base_filename}{output_suffix}.hdf5"
    
    # Save merged dataset
    try:
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('set_x', data=merged_features)
            f.create_dataset('set_y', data=merged_labels)
            
            # Save flow_ids if we have them
            if all_flow_ids:
                merged_flow_ids = np.hstack(all_flow_ids)
                f.create_dataset('flow_ids', data=merged_flow_ids)
                print(f"   💾 Saved flow_ids: {len(merged_flow_ids)} entries")
            
        print(f"💾 Saved merged dataset: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving {output_file}: {e}")
        return False

def merge_all_datasets():
    """Merge HDF5 files for all three main datasets"""
    
    print("="*80)
    print("HDF5 DATASET MERGER FOR FRAGMENTATION ANALYSIS")
    print("="*80)
    
    datasets = {
        "Original": "./TrafficManipulator-master/DATASETS/FLATTEN-PCAPS",
        "Manipulated": "./TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42", 
        "Fragmented": "./TrafficManipulator-master/DATASETS/FRAGMENTED_42"
    }
    
    attack_type = "00-WebDDoS"  # Focus on WebDDoS for now
    
    results = {}
    
    for dataset_name, base_dir in datasets.items():
        print(f"\n🔄 Processing {dataset_name} dataset...")
        success = merge_hdf5_files(base_dir, attack_type, f"_{dataset_name.lower()}_merged")
        results[dataset_name] = success
    
    print(f"\n" + "="*60)
    print("MERGE SUMMARY")
    print("="*60)
    
    for dataset_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{dataset_name:<15}: {status}")
    
    if all(results.values()):
        print(f"\n🎉 All datasets merged successfully!")
        
        # Now create updated comparison script paths
        print(f"\n📝 Updated file paths for comparison:")
        for dataset_name, base_dir in datasets.items():
            merged_file = f"{base_dir}/{attack_type}/10t-100n-DOS2019-flatten-dataset_{dataset_name.lower()}_merged.hdf5"
            print(f"   {dataset_name}: {merged_file}")
        
        # Test the merged datasets
        print(f"\n🧪 Testing merged datasets...")
        test_merged_datasets()
        
    else:
        failed = [name for name, success in results.items() if not success]
        print(f"\n❌ Failed to merge: {', '.join(failed)}")

def test_merged_datasets():
    """Test the merged datasets by loading them and showing statistics"""
    
    attack_type = "00-WebDDoS"
    datasets = {
        "Original": f"./TrafficManipulator-master/DATASETS/FLATTEN-PCAPS/{attack_type}/10t-100n-DOS2019-flatten-dataset_original_merged.hdf5",
        "Manipulated": f"./TrafficManipulator-master/DATASETS/MANIPULATED-FLATTEN_42/{attack_type}/10t-100n-DOS2019-flatten-dataset_manipulated_merged.hdf5",
        "Fragmented": f"./TrafficManipulator-master/DATASETS/FRAGMENTED_42/{attack_type}/10t-100n-DOS2019-flatten-dataset_fragmented_merged.hdf5"
    }
    
    print(f"\n📊 MERGED DATASET STATISTICS:")
    print("-" * 60)
    
    for name, path in datasets.items():
        try:
            with h5py.File(path, 'r') as f:
                features = f['set_x'][:]
                labels = f['set_y'][:]
                
                unique_labels, counts = np.unique(labels, return_counts=True)
                
                print(f"{name:<15}: {len(features):4d} samples")
                for label, count in zip(unique_labels, counts):
                    class_name = "benign" if label == 0 else "attack"
                    print(f"                  Class {label} ({class_name}): {count:3d}")
                
        except Exception as e:
            print(f"{name:<15}: ❌ Error loading - {e}")
    
    print(f"\n🔧 To use merged datasets, update your comparison script paths to use the '*_merged.hdf5' files")

def main():
    """Main function with command line options"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print(__doc__)
            print("\nUsage:")
            print("  python merge_hdf5_datasets.py           # Merge all datasets")
            print("  python merge_hdf5_datasets.py --test    # Test merged datasets")
            print("  python merge_hdf5_datasets.py --help    # Show this help")
            return
        elif sys.argv[1] == "--test":
            test_merged_datasets()
            return
    
    merge_all_datasets()

if __name__ == "__main__":
    main()

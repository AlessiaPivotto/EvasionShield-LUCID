#!/usr/bin/env python3

# Copyright (c) 2022 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Project: LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import h5py
import numpy as np
import os
import glob
import argparse
from collections import defaultdict
from util_functions import ATTACK_CLASSES

def get_attack_type_from_filename(filename):
    """Extract attack type from filename"""
    filename_lower = filename.lower()
    
    if 'webddos' in filename_lower or '00-webddos' in filename_lower:
        return 'WebDDoS'
    elif 'ldap' in filename_lower or '01-ldap' in filename_lower:
        return 'LDAP'
    elif 'portmap' in filename_lower or '02-portmap' in filename_lower:
        return 'Portmap'
    elif 'dns' in filename_lower and 'webddos' not in filename_lower:
        return 'DNS'
    elif 'udplag' in filename_lower or '04-udplag' in filename_lower:
        return 'UDPLag'
    elif 'ntp' in filename_lower or '05-ntp' in filename_lower:
        return 'NTP'
    elif 'snmp' in filename_lower or '06-snmp' in filename_lower:
        return 'SNMP'
    elif 'ssdp' in filename_lower or '07-ssdp' in filename_lower:
        return 'SSDP'
    elif 'syn' in filename_lower or '08-syn' in filename_lower:
        return 'Syn'
    elif 'tftp' in filename_lower or '09-tftp' in filename_lower:
        return 'TFTP'
    elif ('udp' in filename_lower or '10-udp' in filename_lower) and 'udplag' not in filename_lower:
        return 'UDP'
    elif 'netbios' in filename_lower or '11-netbios' in filename_lower:
        return 'NetBIOS'
    elif 'mssql' in filename_lower or '12-mssql' in filename_lower:
        return 'MSSQL'
    else:
        return 'benign'

def modify_dataset_labels(input_folder, output_folder=None):
    """
    Modify HDF5 dataset files to use multiclass labels instead of binary labels
    """
    if output_folder is None:
        output_folder = input_folder + "_multiclass"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Find all HDF5 files
    hdf5_files = glob.glob(os.path.join(input_folder, "**/*100n*.hdf5"), recursive=True)
    
    label_stats = defaultdict(int)
    
    for file_path in hdf5_files:
        print(f"Processing: {file_path}")
        
        # Get relative path to maintain folder structure
        rel_path = os.path.relpath(file_path, input_folder)
        output_path = os.path.join(output_folder, rel_path)
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract attack type from filename/path
        filename = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))
        
        attack_type = get_attack_type_from_filename(filename)
        if attack_type == 'benign':
            attack_type = get_attack_type_from_filename(folder_name)
        
        new_label = ATTACK_CLASSES[attack_type]
        
        print(f"  Attack type: {attack_type} -> Label: {new_label}")
        
        # Read original dataset
        with h5py.File(file_path, 'r') as f_in:
            X = f_in['set_x'][:]
            Y = f_in['set_y'][:]
            
            # Modify labels based on attack type
            if attack_type == 'benign':
                Y_new = np.zeros_like(Y, dtype=np.int32)  # benign = 0
            else:
                # For attack files, change all attack labels (Y > 0) to the specific attack class
                Y_new = np.where(Y > 0, new_label, 0).astype(np.int32)
            
            # Count labels for statistics
            unique, counts = np.unique(Y, return_counts=True)
            for label, count in zip(unique, counts):
                if label == 0:  # benign
                    label_stats['benign'] += count
                else:  # attack
                    label_stats[attack_type] += count
            
            # Save modified dataset
            with h5py.File(output_path, 'w') as f_out:
                f_out.create_dataset('set_x', data=X)
                f_out.create_dataset('set_y', data=Y_new)
        
        print(f"  Saved to: {output_path}")
    
    print("\nLabel Statistics:")
    for attack_type, count in label_stats.items():
        print(f"  {attack_type}: {count} samples")
    
    print(f"\nTotal files processed: {len(hdf5_files)}")
    print(f"Output folder: {output_folder}")

def merge_multiclass_datasets(input_folders, output_folder, dataset_types=['train', 'val', 'test']):
    """
    Merge multiple multiclass datasets into balanced combined datasets
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for dataset_type in dataset_types:
        print(f"\nProcessing {dataset_type} datasets...")
        
        all_X = []
        all_Y = []
        
        for folder in input_folders:
            pattern = os.path.join(folder, f"**/*{dataset_type}.hdf5")
            files = glob.glob(pattern, recursive=True)
            
            # Also try .h5 extension
            if not files:
                pattern = os.path.join(folder, f"**/*{dataset_type}.h5")
                files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                print(f"  Loading: {file_path}")
                
                with h5py.File(file_path, 'r') as f:
                    X = f['set_x'][:]
                    Y = f['set_y'][:]
                    
                    all_X.append(X)
                    all_Y.append(Y)
        
        if all_X:
            # Combine all data
            combined_X = np.vstack(all_X)
            combined_Y = np.hstack(all_Y)
            
            print(f"  Combined {dataset_type} shape: {combined_X.shape}")
            print(f"  Label distribution: {np.bincount(combined_Y)}")
            
            # Save combined dataset
            output_file = os.path.join(output_folder, f"multiclass_{dataset_type}.hdf5")
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('set_x', data=combined_X)
                f.create_dataset('set_y', data=combined_Y)
            
            print(f"  Saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Modify dataset labels for multiclass classification')
    parser.add_argument('input_folder', help='Input folder containing HDF5 files')
    parser.add_argument('--output_folder', help='Output folder (default: input_folder + "_multiclass")')
    parser.add_argument('--merge_folders', nargs='+', help='Folders to merge into combined multiclass dataset')
    parser.add_argument('--merge_output', help='Output folder for merged dataset')
    
    args = parser.parse_args()
    
    if args.merge_folders and args.merge_output:
        merge_multiclass_datasets(args.merge_folders, args.merge_output)
    else:
        modify_dataset_labels(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()

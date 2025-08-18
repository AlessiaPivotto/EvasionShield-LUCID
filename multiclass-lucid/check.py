import h5py
import glob
import numpy as np
import os

def inspect_hdf5_file(file_path):
    """Inspect the structure of an HDF5 file"""
    print(f"\n{'='*60}")
    print(f"File: {file_path}")
    print(f"{'='*60}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            print(f"Keys in file: {list(f.keys())}")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}")
                    print(f"  Shape: {obj.shape}")
                    print(f"  Data type: {obj.dtype}")
                    if obj.size < 100:  # Show data for small datasets
                        print(f"  Data: {obj[:]}")
                elif isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
            
            print("\nDetailed structure:")
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"Error reading file: {e}")

def inspect_folder(folder_path):
    """Inspect all HDF5 files in a folder"""
    hdf5_files = glob.glob(os.path.join(folder_path, "**/*.hdf5"), recursive=True)
    
    print(f"Found {len(hdf5_files)} HDF5 files in {folder_path}")
    
    for file_path in hdf5_files[:3]:  # Only show first 3 files
        inspect_hdf5_file(file_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        inspect_folder(sys.argv[1])
    else:
        inspect_folder("./DATASETS")
import h5py
import numpy as np
import os

# Base directory
base_dir = '/home/rising/EvasionShield-LUCID/TrafficManipulator-master/DATASETS'

# List of HDF5 files to merge (FRAGMENTED datasets)
hdf5_files = [
    f'{base_dir}/MANIPULATED-FLATTEN_42/00-WebDDoS/"10t-100n-DOS2019-flatten-dataset-val.hdf5"',
    f'{base_dir}/MANIPULATED-FLATTEN_42/01-LDAP/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/02-Portmap/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/03-DNS/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/04-UDPLag/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/05-NTP/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/06-SNMP/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/07-SSDP/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/08-Syn/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/09-TFTP/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/10-UDP/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/11-NetBIOS/10t-100n-DOS2019-flatten-dataset-val.hdf5',
    f'{base_dir}/MANIPULATED-FLATTEN_42/12-MSSQL/10t-100n-DOS2019-flatten-dataset-val.hdf5'
]

# Output merged file (create in the same directory as input files)
output_file = f'{base_dir}/MANIPULATED-FLATTEN_42/10t-100n-DOS2019-flatten-dataset-val.hdf5'

# Check which files exist
existing_files = []
for fname in hdf5_files:
    if os.path.exists(fname):
        existing_files.append(fname)
        print(f"Found: {fname}")
    else:
        print(f"Missing: {fname}")

print(f"\nMerging {len(existing_files)} files...")

if not existing_files:
    print("No files found to merge!")
    exit(1)

# Open output file
with h5py.File(output_file, 'w') as out_f:
    for fname in existing_files:
        print(f"Processing: {os.path.basename(fname)}")
        with h5py.File(fname, 'r') as in_f:
            for name in in_f:
                # If dataset already exists, append data
                if name in out_f:
                    data = in_f[name][:]
                    out_f[name].resize((out_f[name].shape[0] + data.shape[0]), axis=0)
                    out_f[name][-data.shape[0]:] = data
                else:
                    data = in_f[name][:]
                    maxshape = (None,) + data.shape[1:]
                    out_f.create_dataset(name, data=data, maxshape=maxshape, chunks=True)

print(f"\nMerged files into {output_file}")

# Print dataset info
with h5py.File(output_file, 'r') as f:
    print(f"\nDataset contents:")
    for key in f.keys():
        print(f"  {key}: {f[key].shape}")
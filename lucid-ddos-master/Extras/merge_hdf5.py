import h5py
import numpy as np


def merge_hdf5_files(*file_list, output_file="merged.hdf5"):
    if len(file_list) < 2:
        raise ValueError("At least two input files are required to merge.")
    # Open all files and collect keys
    with h5py.File(file_list[0], "r") as f0:
        keys = list(f0.keys())
    merged_data = {key: [] for key in keys}
    # Read and concatenate data from all files
    for file in file_list:
        with h5py.File(file, "r") as f:
            for key in keys:
                merged_data[key].append(f[key][:])
    # Concatenate along axis 0 for each key
    for key in keys:
        merged_data[key] = np.concatenate(merged_data[key], axis=0)
    # Write to output file
    with h5py.File(output_file, "w") as f_out:
        for key in keys:
            f_out.create_dataset(key, data=merged_data[key])
    print(f"Saved merged data to {output_file}")

if __name__ == "__main__":

# Merge Train datasets (baseline)
    # merge_hdf5_files("FLAD-PCAPS/00-WebDDoS/10t-100n-DOS2019-dataset-train.hdf5", 
    #                     "FLAD-PCAPS/01-LDAP/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/02-Portmap/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/03-DNS/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/04-UDPLag/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/05-NTP/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/06-SNMP/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/07-SSDP/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/08-Syn/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/09-TFTP/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/10-UDP/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/11-NetBIOS/10t-100n-DOS2019-dataset-train.hdf5",
    #                     "FLAD-PCAPS/12-MSSQL/10t-100n-DOS2019-dataset-train.hdf5"
    #                  )

    merge_hdf5_files("MANIPULATED-PCAPS/00-WebDDoS/10t-100n-DOS2019-dataset-test.hdf5", 
                        "MANIPULATED-PCAPS/01-LDAP/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/02-Portmap/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/03-DNS/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/04-UDPLag/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/05-NTP/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/06-SNMP/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/07-SSDP/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/08-Syn/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/09-TFTP/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/10-UDP/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/11-NetBIOS/10t-100n-DOS2019-dataset-test.hdf5",
                        "MANIPULATED-PCAPS/12-MSSQL/10t-100n-DOS2019-dataset-test.hdf5"
                     )
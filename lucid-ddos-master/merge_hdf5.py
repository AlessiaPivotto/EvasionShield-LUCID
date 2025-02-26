import h5py
import numpy as np


def merge_hdf5_files(file1, file2):
    with h5py.File(file1, "r") as f1, h5py.File(file2, "r") as f2:
        keys = list(f1.keys())
        for key in keys:
            data1 = f1[key][:]
            data2 = f2[key][:]
            merged_data = np.concatenate([data1, data2], axis=0)
            with h5py.File("merged.hdf5", "a") as f:
                f.create_dataset(key, data=merged_data)
        print(f"Saved merged data to merged.hdf5")

if __name__ == "__main__":

    merge_hdf5_files("Benign-Syn-Manipulated/10t-10n-DOS2019-dataset-test.hdf5", "Benign-Syn-Manipulated/10t-10n-DOS2019-dataset-test.hdf5")
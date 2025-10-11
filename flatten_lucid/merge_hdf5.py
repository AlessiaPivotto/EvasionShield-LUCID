import h5py
import glob
import os

def merge_hdf5_files(input_files, output_file):
    with h5py.File(output_file, 'w') as out_f:
        for file_path in input_files:
            with h5py.File(file_path, 'r') as in_f:
                for name in in_f:
                    if name in out_f:
                        # If dataset exists, append data
                        data = in_f[name][:]
                        out_f[name].resize((out_f[name].shape[0] + data.shape[0]), axis=0)
                        out_f[name][-data.shape[0]:] = data
                    else:
                        # Create new dataset with maxshape for appending
                        data = in_f[name][:]
                        out_f.create_dataset(
                            name,
                            data=data,
                            maxshape=(None,) + data.shape[1:],
                            chunks=True
                        )

if __name__ == "__main__":
    # Example: merge all .h5 files in a folder
    input_folder = "./DATASET_RIDOTTO/Manipulated/Manipulated_split"
    output_file = "./DATASET_RIDOTTO/Manipulated/10t-100n-DOS2019-dataset-test.hdf5"
    input_files = sorted(glob.glob(os.path.join(input_folder, "*.hdf5")))
    merge_hdf5_files(input_files, output_file)
    print(f"Merged {len(input_files)} files into {output_file}")
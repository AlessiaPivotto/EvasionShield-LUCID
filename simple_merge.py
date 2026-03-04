import sys
import os
import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Iterable, List

#!/usr/bin/env python3
"""
simple_merge.py

Merge two or more HDF5 files into a single HDF5 file.

Usage (command-line):
    python simple_merge.py src1.h5 src2.h5 -o out.h5 --mode overwrite
    python simple_merge.py a.h5 b.h5 c.h5 -o merged.h5 --mode append

Or use the "preconfigured merge" by editing the lists below and
simply running:
    python simple_merge.py

Modes:
    overwrite - if an object exists in destination it will be replaced by the source.
    skip      - existing objects in destination are left unchanged.
    append    - for datasets with the same dtype and matching trailing shape,
                stack along axis 0 (only when sensible); otherwise falls back
                to overwrite.
"""


# ---------------------------------------------------------------------------
# Preconfigured merge
# ---------------------------------------------------------------------------
# Edit the following lists to run a merge without passing command-line
# arguments. If you run "python simple_merge.py" with no arguments and
# PRECONFIGURED_SOURCES is not empty, these values will be used.

# Example (uncomment and adapt to your paths):
# PRECONFIGURED_SOURCES: List[str] = [
#     "DATASETS/merged_flatten_train_dataset.hdf5",
#     "DATASETS/merged_matrix_10n_train_dataset.hdf5",
#     "DATASETS/merged_matrix_100n_train_dataset.hdf5",
# ]
# PRECONFIGURED_OUTPUT: str = "DATASETS/merged_all_train_datasets.hdf5"

PRECONFIGURED_SOURCES: List[str] = []
PRECONFIGURED_OUTPUT: str = "10t-100n-DOS2019-flatten-dataset-train.hdf5"
PRECONFIGURED_MODE: str = "append"      # "overwrite", "skip", or "append"
PRECONFIGURED_FRESH: bool = True         # True: overwrite existing output file

# When no PRECONFIGURED_SOURCES are set, the script can automatically
# discover all "*-train.hdf5" files under this root directory and merge
# them. This matches your request to merge all *-train.hdf5 files that
# live inside subfolders under DATASETS/MANIPULATED-FLATTEN_42.
AUTO_SEARCH_ROOT: Path = Path("DATASETS/MANIPULATED-FLATTEN_42")
AUTO_PATTERN: str = "*-train.hdf5"



def copy_attrs(src, dst):
    for k, v in src.attrs.items():
        try:
            dst.attrs[k] = v
        except Exception:
            # skip attributes that can't be copied
            pass


def merge_group(src_group, dst_group, mode="overwrite"):
    """
    Recursively merge src_group into dst_group.

    mode: "overwrite", "skip", "append"
    """
    copy_attrs(src_group, dst_group)

    for name, obj in src_group.items():
        if name in dst_group:
            dst_obj = dst_group[name]
            # Both are groups -> recurse
            if isinstance(obj, h5py.Group) and isinstance(dst_obj, h5py.Group):
                merge_group(obj, dst_obj, mode)
                continue

            # Both are datasets
            if isinstance(obj, h5py.Dataset) and isinstance(dst_obj, h5py.Dataset):
                if mode == "skip":
                    continue
                if mode == "overwrite":
                    # remove existing and copy
                    try:
                        del dst_group[name]
                    except Exception:
                        pass
                    src_group.copy(name, dst_group, name=name)
                    # ensure attributes copied
                    copy_attrs(obj, dst_group[name])
                    continue
                if mode == "append":
                    # Try to append along axis 0 if trailing shapes match
                    ds_src = obj
                    ds_dst = dst_obj
                    # Fast checks
                    try:
                        if ds_src.dtype == ds_dst.dtype and ds_src.ndim >= 1 and ds_dst.ndim >= 1 and ds_src.shape[1:] == ds_dst.shape[1:]:
                            # Read into memory and concatenate
                            a = ds_dst[()]
                            b = ds_src[()]
                            concatenated = np.concatenate([a, b], axis=0)
                            # Replace dst dataset with concatenated
                            del dst_group[name]
                            dst_group.create_dataset(name, data=concatenated, dtype=concatenated.dtype,
                                                     compression=getattr(ds_src, "compression", None),
                                                     chunks=getattr(ds_src, "chunks", None))
                            copy_attrs(obj, dst_group[name])
                            continue
                        else:
                            # cannot append, fallback to overwrite
                            del dst_group[name]
                            src_group.copy(name, dst_group, name=name)
                            copy_attrs(obj, dst_group[name])
                            continue
                    except Exception:
                        # fallback to overwrite on any error
                        try:
                            del dst_group[name]
                        except Exception:
                            pass
                        src_group.copy(name, dst_group, name=name)
                        copy_attrs(obj, dst_group[name])
                        continue

            # Types differ (one is group, other dataset) -> behavior depends on mode
            if mode == "skip":
                continue
            # overwrite by default for mismatched types
            try:
                del dst_group[name]
            except Exception:
                pass
            src_group.copy(name, dst_group, name=name)
            copy_attrs(obj, dst_group[name])
        else:
            # doesn't exist in dest -> copy directly
            src_group.copy(name, dst_group, name=name)
            copy_attrs(obj, dst_group[name])


def merge_files(sources: Iterable[str], destination: str, mode="overwrite", overwrite_dest=False):
    """
    Merge a list of source HDF5 files into destination HDF5 file.

    If overwrite_dest is False and destination exists, sources are merged into it.
    If overwrite_dest is True, an existing destination file will be removed / replaced.
    """
    if overwrite_dest:
        # open destination with 'w' to start fresh
        with h5py.File(destination, "w") as dst:
            # empty file created
            pass

    # open destination in append mode
    with h5py.File(destination, "a") as dst:
        for src_path in sources:
            if src_path == destination:
                # skip merging file into itself
                continue
            with h5py.File(src_path, "r") as src:
                merge_group(src, dst, mode=mode)


def discover_train_files(root: Path, pattern: str = AUTO_PATTERN) -> List[str]:
    """Recursively discover all HDF5 train files under root matching pattern.

    Returns an alphabetically sorted list of file paths as strings.
    """
    if not root.exists():
        return []

    files = [
        str(p)
        for p in sorted(root.rglob(pattern))
        if p.is_file()
    ]
    return files


def parse_args(argv):
    p = argparse.ArgumentParser(description="Merge HDF5 files.")
    p.add_argument("sources", nargs="+", help="Source HDF5 files to merge.")
    p.add_argument("-o", "--output", required=True, help="Destination HDF5 file.")
    p.add_argument("--mode", choices=("overwrite", "skip", "append"), default="overwrite",
                   help="Conflict resolution mode (default: overwrite).")
    p.add_argument("--fresh", action="store_true", help="Create fresh output file (overwrite existing).")
    return p.parse_args(argv)


def main(argv):
    # If command-line arguments are provided, use them (original behavior).
    if argv:
        args = parse_args(argv)
        merge_files(args.sources, args.output, mode=args.mode, overwrite_dest=args.fresh)
        return

    # No command-line arguments: either use explicitly configured sources
    # or automatically discover all *-train.hdf5 files under AUTO_SEARCH_ROOT.
    if PRECONFIGURED_SOURCES:
        sources = PRECONFIGURED_SOURCES
    else:
        sources = discover_train_files(AUTO_SEARCH_ROOT)

    if not sources:
        print("No command-line arguments provided and no sources found.\n"
              "Either set PRECONFIGURED_SOURCES in simple_merge.py, or create\n"
              f"HDF5 files matching '{AUTO_PATTERN}' under '{AUTO_SEARCH_ROOT}'.")
        return

    # Place the output file inside AUTO_SEARCH_ROOT unless PRECONFIGURED_OUTPUT
    # already includes a directory or an absolute path.
    output_path = PRECONFIGURED_OUTPUT
    if not os.path.isabs(output_path) and not os.path.dirname(output_path):
        output_path = str(AUTO_SEARCH_ROOT / output_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("Running automatic train-merge...")
    print("  Sources:")
    for src in sources:
        print(f"    - {src}")
    print(f"  Output : {output_path}")
    print(f"  Mode   : {PRECONFIGURED_MODE} (fresh={PRECONFIGURED_FRESH})")

    merge_files(sources, output_path,
                mode=PRECONFIGURED_MODE, overwrite_dest=PRECONFIGURED_FRESH)


if __name__ == "__main__":
    main(sys.argv[1:])
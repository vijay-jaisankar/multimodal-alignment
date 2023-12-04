"""
    Separate out the files containing 1024 points per cloud - this will serve as a list of candidates to select a sample for minimal testing for choosing a zSC.
"""
import pickle
import numpy as np
import sys

# Path to the filename:num_points mapping
orig_mapping_path = "../load-modelnet/dataset_portion_size_mapping.pkl"

# Path to the individual .npy files
npy_files_path = "/home/vijay/Desktop/college/active-work/multimodal-alignment/data/modelnet40c-numpy/modelnet40_c"

# Load the original mapping
with open(orig_mapping_path, "rb") as f:
    data = pickle.load(f)

if data is None:
    sys.exit(1)

# Filter out 1024 points' filenames
data_filtered = [
    x for x in data
    if data[x] == 1024
]

# Append paths and store the filtered list
data_filtered = [
    f"{npy_files_path}/{x}"
    for x in data_filtered
]

with open("candidate_files_list.pkl", "wb") as f:
    pickle.dump(data_filtered, f)

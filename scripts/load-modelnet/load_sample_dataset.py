"""
    Load a sample portion of the Modelnet-40C dataset and run some diagnostics
"""

import numpy as np
import glob
from labels_value_counts import base_dir_path, load_npy
from tqdm import tqdm
import pickle

dataset_size_mapping = {}

if __name__ == "__main__":
    sample_dataset_file = f"{base_dir_path}/data_background_1.npy"
    data = load_npy(sample_dataset_file)
    
    # Load all npy files
    all_npy_files = (glob.glob(f"{base_dir_path}/*.npy")) 
    all_npy_files = [x for x in all_npy_files if "label"  not in x] # Exclude the labels' file
    
    for f in tqdm(all_npy_files):
        data = load_npy(f)
        if data is not None:
            num_points_per_cloud = data.shape[1]
            print(f"Filename: {f}, number of points per cloud: {num_points_per_cloud}")
            
            dataset_portion = f.split("/")[-1]
            if dataset_portion not in dataset_size_mapping:
                dataset_size_mapping[dataset_portion] = num_points_per_cloud


    # Save the pickle file
    with open("dataset_portion_size_mapping.pkl", "wb") as fp:
        pickle.dump(dataset_size_mapping, fp)

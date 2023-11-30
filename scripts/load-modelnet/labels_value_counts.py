"""
    Show the value counts of the labels' file
"""

import numpy as np

# Base directory path for the .npy files
base_dir_path = "/home/vijay/Desktop/college/active-work/multimodal-alignment/data/modelnet40c-numpy/modelnet40_c"

# Load npy file
def load_npy(filename):
    data = None
    try:
        data =  np.load(filename, allow_pickle=True) 
    except Exception as e:
        print(f"Error while loading {filename}: {e}")
        return None
    return data

if __name__ == "__main__":
    # Load the labels file
    label_filename = f"{base_dir_path}/label.npy"
    data = np.array(load_npy(label_filename))
    print(f"Shape of the data file: {data.shape}")

    # Value counts
    value_counts = np.array(np.unique(data, return_counts=True)).T
    print(value_counts)

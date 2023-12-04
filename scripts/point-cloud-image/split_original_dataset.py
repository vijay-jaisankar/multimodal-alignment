"""
    Convert the original 2468*1024*3 dataset into individual 1*1024*3 point clouds labelled accordingly
"""

import numpy as np
from tqdm import tqdm

# Load numpy file
def load_np(file_name):
    data = np.load(file_name)
    return data

# Save numpy file
def save_np(x, file_name):
    with open(file_name, "wb") as f:
        np.save(f, x)

# Location of the numpy files
ORIG_NP_PATH = "/home/vijay/Desktop/college/active-work/multimodal-alignment/data/modelnet40c-numpy/filtered_np"

# Load labels and the `data_original.npy`
labels = load_np(f"{ORIG_NP_PATH}/label.npy")
data = load_np(f"{ORIG_NP_PATH}/data_original.npy")

# Map of the number of examples encountered so far
freq_map = {}

# Iterate through the dataset
N = len(data)

for i in tqdm(range(N)):
    # Point cloud: 1024x3
    x = data[i]

    # Reshape to 1x1024x3
    pcl = np.array([x])

    # Label Index
    label_index = int(labels[i][0])

    # Update frequencies for the label
    if label_index not in freq_map:
        freq_map[label_index] = 1
    else:
        freq_map[label_index] += 1

    # Construct file name for the point cloud
    output_file_name = f"point_cloud_class_{label_index}_example_{freq_map[label_index]}.npy"

    # Save individual point cloud
    save_np(pcl, output_file_name)


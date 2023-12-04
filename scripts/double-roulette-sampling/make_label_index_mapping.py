"""
    Create and save a mapping of the form
    LABEL:INDICES for the ModelNet40-C dataset
"""
import pickle
import numpy as np

# Parent mapping
mapping = {}

# Labels' file path
labels_file_path = "/home/vijay/Desktop/college/active-work/multimodal-alignment/data/modelnet40c-numpy/modelnet40_c/label.npy"

# Pre-populate the mapping
for i in range(0, 40):
    mapping[i] = []

# Load the labels' file
labels_data = np.load(labels_file_path)

# Iterate through the labels
N = len(labels_data)
for i in range(N):
    current_index = int(i)
    current_label = int(labels_data[i][0])
    mapping[current_label].append(current_index)

# Save the mapping
with open("label_indices_mapping.pkl", "wb") as f:
    pickle.dump(mapping, f)


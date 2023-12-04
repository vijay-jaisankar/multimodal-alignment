"""
    Double Roulette Sampling: Pipeline
        - For a class label, choose an index for the point cloud
        - For an index, choose a file name to extract the point cloud
        - Store individual np files indexed by class label
"""
import numpy as np
import pickle
import random
import sys
import gc

# Set random seed
random.seed(42)

# Load the label:indices mapping
with open("label_indices_mapping.pkl", "rb") as f:
    label_indices_mapping = pickle.load(f)

if label_indices_mapping is None:
    print("Label:Indices file read error")
    sys.exit(1)

# Load the candidate files' list
with open("candidate_files_list.pkl", "rb") as f:
    candidate_files_list = pickle.load(f)

if candidate_files_list is None:
    print("Candidate files' read error")
    sys.exit(1)

candidate_files_list = np.array(candidate_files_list)

# Get random index for class label
def get_random_pcl_index(class_label:int) -> int:
    # Sanity check
    if class_label < 0 or class_label >= 40:
        return -1
    
    # Extract the candidate list of indices
    global label_indices_mapping
    X = np.array(label_indices_mapping[class_label])
    
    # Random selection for global chosen index
    chosen_index_local = np.random.choice(X.shape[0], 1, replace = False) # ref: https://stackoverflow.com/questions/43506766/randomly-select-from-numpy-array
    global_chosen_index = int(X[chosen_index_local[0]])
    return global_chosen_index

# Instead of wasting RNG calls for the filenames, simply permute the list and take the first 40 elements - this also prevents bias/duplication
candidate_files_list = np.random.permutation(candidate_files_list)[:40]

# Iterate through the process for all 40 classes
for class_label in range(0, 40):
    # Get chosen index and file name
    chosen_index = get_random_pcl_index(class_label)
    chosen_file_name = candidate_files_list[class_label]

    # Get point cloud
    data = np.load(chosen_file_name)
    x = data[chosen_index]
    pcl = np.array([x])

    # Save individual point cloud
    output_file_name = f"testmin_{class_label}.npy"
    with open(output_file_name, "wb") as f:
        np.save(f, pcl)

    # Clean up
    data = None
    pcl = None
    _ = gc.collect()

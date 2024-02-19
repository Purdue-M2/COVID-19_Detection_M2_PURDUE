import h5py
import numpy as np
import torch


file_path1 = 'all_train_teach.h5'
file_path2 = 'task2_non_clip_vit.h5'
combined_file_path = 'all_train_stu.h5'

# Open the source HDF5 files
with h5py.File(file_path1, 'r') as h5f1, h5py.File(file_path2, 'r') as h5f2:
    # Load the datasets and convert them to PyTorch tensors if they aren't already
    data1 = torch.tensor(h5f1['train_features'][:], dtype=torch.float32)
    data2 = torch.tensor(h5f2['train_features'][:], dtype=torch.float32)
    
    # Combine the data
    combined_data = torch.cat([data1, data2], dim=0)

# Create a new HDF5 file for the combined dataset
with h5py.File(combined_file_path, 'w') as h5f_combined:
    # Create a dataset in the combined file. Convert the tensor to a NumPy array.
    h5f_combined.create_dataset('train_features', data=combined_data.numpy(), dtype='float32')

print(f"Combined dataset shape: {combined_data.shape}")
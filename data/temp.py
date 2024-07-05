import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import argparse
import random

parser = argparse.ArgumentParser(description='Process .npy files in subdirectories.')
parser.add_argument('--chunk_dir', help='Directory containing chunks files')
parser.add_argument('--sub_dir', help='Subdirectory to process')
parser.add_argument('--num_files', type=int, default=100, help='Number of files to process (default: 100)')
parser.add_argument('--random', action='store_true', help='Randomly select files instead of taking the first N')
args = parser.parse_args()

chunkdir = args.chunk_dir
subdir = args.sub_dir
num_files = args.num_files
use_random = args.random

# Get all .npy files from the specified directory
all_files = glob(f'{chunkdir}/{subdir}/*.npy')

# Select files based on the random parameter
if use_random:
    file_list = random.sample(all_files, min(num_files, len(all_files)))
else:
    file_list = all_files[:num_files]

# Initialize list to store channel data
channel_data = [[] for _ in range(6)]  # Now 6 channels

# Load and process each file
for file in file_list:
    print(f"Processing file: {os.path.basename(file)}")
    patch = np.load(file)
    for i in range(6):
        channel_values = patch[:,:,i].flatten()
        channel_data[i].extend(channel_values)

# Convert lists to numpy arrays for easier computation
channel_data = [np.array(channel) for channel in channel_data]

# Create subplots for each channel
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
colors = ['red', 'green', 'blue', 'red', 'green', 'blue']
channel_names = ['red.tif', 'reg.tif', 'raster.tif (dem)', 'Red', 'Green', 'Blue']

# Plot distributions for each channel
for i in range(6):
    row, col = divmod(i, 3)
    unique_freq, counts = np.unique(channel_data[i], return_counts=True)
    axs[row, col].plot(unique_freq, counts, color=colors[i], alpha=0.7)
    axs[row, col].set_title(f'{channel_names[i]} Distribution')
    axs[row, col].set_xlabel('Value')
    axs[row, col].set_ylabel('Count')
   
    min_val = np.min(channel_data[i])
    max_val = np.max(channel_data[i])
    axs[row, col].text(0.05, 0.95, f'Min: {min_val:.2f}\nMax: {max_val:.2f}',
                       transform=axs[row, col].transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()
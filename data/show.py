import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# usage: python show.py --orchard_id 1676

parser = argparse.ArgumentParser(description='Process .npy files in subdirectories.')
parser.add_argument('--chunk_dir', default='chunks/', help='Base directory where the .npy files are saved')
parser.add_argument('--orchard_id', default='', help='Orchard ID')
args = parser.parse_args()

# set the base output directory and orchard id
chunk_dir = args.chunk_dir
orchard_id = args.orchard_id

# define the subdirectories
subdirs = ["case_1", "case_2", "case_3", "normal"]

# Loop through all subdirectories in the base output directory
for subdir in subdirs:
    output_dir = os.path.join(chunk_dir, subdir)
    print(f"Checking directory: {output_dir}")
    
    # Check if the directory exists before trying to list its contents
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist.")
        continue
    
    # Loop through all files in each subdirectory
    for filename in os.listdir(output_dir):
        if filename.endswith(".npy") and orchard_id in filename:
            # Load the NumPy array from the file
            filepath = os.path.join(output_dir, filename)
            block_array = np.load(filepath)
            block_array = block_array.transpose(2, 0, 1)
            
            # Determine the number of bands in the array
            num_bands = block_array.shape[0]
            
            # Create a figure with subplots for each band
            fig, axes = plt.subplots(1, num_bands, figsize=(2*num_bands, 3))
            fig.suptitle(f"{subdir}/{filename}")
            
            # Plot each band as a subplot
            for band in range(num_bands):
                if num_bands == 1:
                    ax = axes
                else:
                    ax = axes[band]
                ax.imshow(block_array[band], cmap='gray')
                ax.set_title(f"Band {band + 1}")
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
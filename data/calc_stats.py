import os
import sys
import numpy as np
from typing import List
import argparse

def calc_mean(chunk_path: str, num_channels: int):
    print("Calculating mean")
    pixel_count = 0
    channel_sums = np.zeros(shape=(num_channels,), dtype=np.float64)

    for root, dirs, files in os.walk(chunk_path):
        if os.path.basename(root) in ["normal", "case_1", "case_2"]:
            for file in files:
                file_path = os.path.join(root, file)
                file_data = np.load(file_path)
                channel_sums += np.sum(file_data, axis=(0, 1))
                pixel_count += file_data.shape[0] * file_data.shape[1]

    return channel_sums / pixel_count

def calc_sdv(means: List[int], chunk_path: str, num_channels: int):
    print("Calculating standard deviation")
    pixel_count = 0
    channel_sums = np.zeros(shape=(num_channels,), dtype=np.float64)

    for root, dirs, files in os.walk(chunk_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_data = np.load(file_path)
                
                for channel in range(num_channels):
                    squared_diff = (file_data[:, :, channel] - means[channel]) ** 2
                    channel_sums[channel] += np.sum(squared_diff)

                pixel_count += file_data.shape[0] * file_data.shape[1]

    return np.sqrt(channel_sums / pixel_count)

# could become problematic for large amounts of data (maybe...)
def calc_percentiles(chunk_path: str, num_channels: int, percentiles: List[int]):
    print("Calculating percentiles")

    result = np.zeros((num_channels, len(percentiles)), dtype=np.float64)
    # read for each channel individually to save memory
    for channel in range(num_channels):
        channel_data = []
        for root, dirs, files in os.walk(chunk_path):
            if os.path.basename(root) in ["normal", "case_1", "case_2"]:

                for file in files:
                    file_path = os.path.join(root, file)
                    file_data = np.load(file_path)
                    # for the current channel extract the data for each chunk, flatten and append to the list
                    channel_data.extend(file_data[:, :, channel].flatten())
        
        result[channel, :] = np.percentile(channel_data, percentiles)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the mean and standard deviation of each channel in our dataset")
    parser.add_argument("chunk_path", type=str, help="The path to the directory containing the dataset")
    parser.add_argument("num_channels", type=int, help="The number of channels in the dataset")
    args = parser.parse_args()
    path = args.chunk_path
    num_channels = args.num_channels

    train_means = calc_mean(path, num_channels)
    train_sdv = calc_sdv(train_means, path, num_channels)
    percentiles = calc_percentiles(path, num_channels, [1, 99])
    
    # write to file in data_stats folder
    # make data_stats folder if it doesn't exist
    if not os.path.exists("data_stats"):
        os.makedirs("data_stats")
    
    mean_path = os.path.join("data_stats", "train_means.npy")
    sdv_path = os.path.join("data_stats", "train_sdv.npy")
    percentiles_path = os.path.join("data_stats", "train_percentiles.npy")
    
    np.save(mean_path, train_means)
    print("Means for each channel: ", train_means)
    np.save(sdv_path, train_sdv)
    print("Standard Deviation for each channel: ", train_sdv)
    np.save(percentiles_path, percentiles)
    print("Percentiles (1st and 99th) for each channel:", percentiles)
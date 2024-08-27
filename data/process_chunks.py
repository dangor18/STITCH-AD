import os
import numpy as np
import concurrent.futures
import multiprocessing
import random
import shutil
import argparse
from typing import List, Tuple

def is_orchard_folder(folder_path: str) -> bool:
    """Check if the folder is an orchard folder containing required subfolders."""
    required_folders = ['normal', 'case_1', 'case_2', 'case_3']
    return all(os.path.isdir(os.path.join(folder_path, subfolder)) for subfolder in required_folders)

def train_test_split(folder_path: str):
    """Perform train/test split on the data."""
    print(f"Performing train/test split for: {folder_path}")
    
    for dir_name in ['train', 'test']:
        if os.path.exists(os.path.join(folder_path, dir_name)):
            shutil.rmtree(os.path.join(folder_path, dir_name))
    
    os.makedirs(os.path.join(folder_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'test', 'normal'), exist_ok=True)
    
    if os.path.exists(os.path.join(folder_path, 'normal')):
        shutil.move(os.path.join(folder_path, 'normal'), os.path.join(folder_path, 'train', 'normal'))
    
    case_counts = []
    for case in ['case_1', 'case_2', 'case_3']:
        case_path = os.path.join(folder_path, case)
        if os.path.exists(case_path):
            if not os.listdir(case_path):
                print(f"{case} is empty. Removing it.")
                shutil.rmtree(case_path)
            else:
                shutil.move(case_path, os.path.join(folder_path, 'test', case))
                npy_count = len([f for f in os.listdir(os.path.join(folder_path, 'test', case)) if f.endswith('.npy')])
                case_counts.append(npy_count)
                print(f"{case} contains {npy_count} .npy files")
    
    if case_counts:
        avg_count = sum(case_counts) // len(case_counts)
        print(f"Average .npy count: {avg_count}")
        normal_files = [f for f in os.listdir(os.path.join(folder_path, 'train', 'normal')) if f.endswith('.npy')]
        files_to_move = random.sample(normal_files, min(avg_count, len(normal_files)//4))
        
        for file in files_to_move:
            shutil.move(os.path.join(folder_path, 'train', 'normal', file), 
                        os.path.join(folder_path, 'test', 'normal', file))
        
        print(f"Moved {len(files_to_move)} .npy files from train/normal to test/normal")
    else:
        print("No non-empty case folders found. No files moved to test/normal.")

def main(root_dir: str):
    orchards = [orchard for orchard in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, orchard)) and 
                is_orchard_folder(os.path.join(root_dir, orchard))]
    
    # split each orchards data (given the orchards data directory is present in the root directory)
    for orchard in orchards:
        orchard_path = os.path.join(root_dir, orchard)
        # split the data into a train and test set
        train_test_split(orchard_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and scale image data for orchards.")
    parser.add_argument("root_dir", type=str, help="Path to the root directory containing the orchard folders")
    args = parser.parse_args()
    
    main(args.root_dir)
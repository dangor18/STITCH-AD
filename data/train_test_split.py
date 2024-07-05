import os
import random
import math
import sys

def organize_folders(base_dir):
    os.chdir(base_dir)
    
    os.makedirs('train', exist_ok=True)
    os.makedirs('val', exist_ok=True)
    os.makedirs('val/normal', exist_ok=True)

    if os.path.exists('normal'):
        os.rename('normal', 'train/normal')

    case_counts = []
    for case in ['case_1', 'case_2']:
        if os.path.exists(case):
            os.rename(case, f'val/{case}')
            npy_count = len([f for f in os.listdir(f'val/{case}') if f.endswith('.npy')])
            case_counts.append(npy_count)
            print(f"{case} contains {npy_count} .npy files")

    if case_counts:
        avg_count = sum(case_counts) 
        print(f"Average .npy count: {avg_count}")
        normal_files = [f for f in os.listdir('train/normal') if f.endswith('.npy')]
        files_to_move = random.sample(normal_files, min(avg_count, len(normal_files)//4))

        for file in files_to_move:
            os.rename(os.path.join('train/normal', file), os.path.join('val/normal', file))

        print(f"Moved {len(files_to_move)} .npy files from train/normal to val/normal")
    else:
        print("No case folders found. No files moved to val/normal.")

    print("Folders reorganized successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py /path/to/directory")
        sys.exit(1)
    
    base_directory = sys.argv[1]
    if not os.path.isdir(base_directory):
        print(f"Error: {base_directory} is not a valid directory")
        sys.exit(1)

    organize_folders(base_directory)
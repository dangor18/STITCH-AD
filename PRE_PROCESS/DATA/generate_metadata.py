import os
import json
import numpy as np
import argparse
import random
from collections import defaultdict
import shutil

# Define the class names and their corresponding labels
class_labels = {
   "normal": 0,
   "case_1": 1,
   "case_2": 1,
   "case_3": 1
}

def calculate_orchard_minmax(orchard_path):
   """
   Calculate min/max values for all chunks in an orchard.
   
   Args:
       orchard_path (str): Path to the orchard directory.

   Returns:
       tuple: (min_vals, max_vals) with min/max values for normalization
   """
   min_vals = None
   max_vals = None
   
   # Process both train and test directories
   for split in ["train", "test"]:
       split_path = os.path.join(orchard_path, split)
       if not os.path.exists(split_path):
           continue
           
       # Process all classes in the split
       for class_name in class_labels.keys():
           class_path = os.path.join(split_path, class_name)
           if not os.path.exists(class_path):
               continue
               
           # Process each .npy file
           for filename in os.listdir(class_path):
               if filename.endswith(".npy"):
                   data = np.load(os.path.join(class_path, filename))
                   curr_min = np.percentile(data, 1, axis=(0,1))
                   curr_max = np.percentile(data, 99, axis=(0,1))
                   
                   if min_vals is None:
                       min_vals = curr_min
                       max_vals = curr_max
                   else:
                       min_vals = np.minimum(min_vals, curr_min)
                       max_vals = np.maximum(max_vals, curr_max)
   
   return min_vals.tolist(), max_vals.tolist()

def process_directory(dir_path, class_name, split, orchard_name, min_vals, max_vals):
   """
   Process a directory and generate metadata for its contents.

   Args:
       dir_path (str): Path to directory to process
       class_name (str): Name of the class (e.g., "normal", "case_1", etc.)
       split (str): Dataset split (e.g., "train" or "test")
       orchard_name (str): Name of the orchard
       min_vals (list): Minimum values for normalization
       max_vals (list): Maximum values for normalization

   Returns:
       list: List of metadata entries for files in the directory
   """
   metadata = []
   for filename in os.listdir(dir_path):
       if filename.endswith(".npy"):
           relative_path = os.path.join(orchard_name, split, class_name, filename)
           metadata_entry = {
               "filename": relative_path,
               "label": class_labels[class_name],
               "label_name": "good" if class_labels[class_name] == 0 else "defective",
               "clsname": orchard_name,
               "min_vals": min_vals,
               "max_vals": max_vals,
               "case": class_name,
               "x": int(os.path.splitext(os.path.basename(filename))[0].split("_")[-1]),
               "y": int(os.path.splitext(os.path.basename(filename))[0].split("_")[-2])
           }
           metadata.append(metadata_entry)
   return metadata

def main(root_dir, verbose, totals, orchard_level):
   """
   Main function to generate metadata for the orchard dataset.

   Args:
       root_dir (str): Root directory of the dataset
       verbose (bool): If True, print detailed statistics
       totals (bool): If True, only print total statistics
   """
   all_metadata = {"train": [], "test": []}
   case_metadata = defaultdict(lambda: defaultdict(list))
   normal_metadata = defaultdict(list)
   
   # delete metadata folder if it exists
   metadata_dir = os.path.join(root_dir, "metadata")
   if os.path.exists(metadata_dir):
      shutil.rmtree(metadata_dir)

   for orchard in os.listdir(root_dir):
       orchard_path = os.path.join(root_dir, orchard)
       if os.path.isdir(orchard_path):
           min_vals, max_vals = calculate_orchard_minmax(orchard_path)
           
           train_dir = os.path.join(orchard_path, "train")
           if os.path.exists(train_dir):
               normal_dir = os.path.join(train_dir, "normal")
               if os.path.exists(normal_dir):
                   all_metadata["train"].extend(
                       process_directory(normal_dir, "normal", "train", orchard, min_vals, max_vals)
                   )
           
           test_dir = os.path.join(orchard_path, "test")
           if os.path.exists(test_dir):
               for class_name in class_labels.keys():
                   class_dir = os.path.join(test_dir, class_name)
                   if os.path.exists(class_dir):
                       processed_data = process_directory(
                           class_dir, class_name, "test", orchard, min_vals, max_vals
                       )
                       if class_name != "normal":
                           case_metadata[class_name][orchard].extend(processed_data)
                       else:
                           normal_metadata[orchard].extend(processed_data)

   ordered_test_metadata = []

   # Collect statistics for each case and all cases
   statistics = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
   for case_name in sorted(case_metadata.keys()):
       all_case_data = []
       for orchard, case_data in case_metadata[case_name].items():
           if orchard in normal_metadata and normal_metadata[orchard]:
               needed_normal = len(case_data)
               sampled_normal = random.sample(normal_metadata[orchard], min(needed_normal, len(normal_metadata[orchard])))
               all_case_data.extend(case_data)
               all_case_data.extend(sampled_normal)
               
               statistics[case_name][orchard]['normal'] = len(sampled_normal)
               statistics[case_name][orchard]['defective'] = len(case_data)
               statistics[f'all_{case_name}'][orchard]['normal'] += len(sampled_normal)
               statistics[f'all_{case_name}'][orchard]['defective'] += len(case_data)

       if all_case_data:
           all_case_clsname = f"all_{case_name}"
           for entry in all_case_data:
               new_entry = entry.copy()
               new_entry["clsname"] = all_case_clsname
               ordered_test_metadata.append(new_entry)

   if not totals:
       # Then, add orchard_case_x entries
       for case_name in sorted(case_metadata.keys()):
           for orchard in sorted(case_metadata[case_name].keys()):
               case_data = case_metadata[case_name][orchard]
               if orchard in normal_metadata and normal_metadata[orchard]:
                   needed_normal = len(case_data)
                   sampled_normal = random.sample(normal_metadata[orchard], min(needed_normal, len(normal_metadata[orchard])))
                   
                   orchard_case_clsname = f"{orchard}_{case_name}"
                   for entry in case_data + sampled_normal:
                       new_entry = entry.copy()
                       new_entry["clsname"] = orchard_case_clsname
                       ordered_test_metadata.append(new_entry)
               else:
                   print(f"Warning: No normal samples for orchard {orchard}. Skipping {case_name} for this orchard.")

   # Print statistics if in verbose mode
   if verbose:
       print("\nStatistics for each case:")
       for case_name in sorted(statistics.keys()):
           print(f"\n{case_name.upper()}:")
           for orchard in sorted(statistics[case_name].keys()):
               print(f"  {orchard}:")
               print(f"    Normal: {statistics[case_name][orchard]['normal']}")
               print(f"    Defective: {statistics[case_name][orchard]['defective']}")

   # Create metadata folder if it doesn't exist
   metadata_dir = os.path.join(root_dir, "metadata")
   if not os.path.exists(metadata_dir):
       os.makedirs(metadata_dir)

   if orchard_level:
    all_metadata_unique = generate_all_metadata(ordered_test_metadata, all_metadata)

    # Save the all unique metadata to a JSON file
    all_output_file = os.path.join(metadata_dir, "all_metadata.json")  # New output file
    with open(all_output_file, "w") as f:
            for entry in all_metadata_unique:
                json.dump(entry, f)
                f.write("\n")

    print(f"All metadata file '{all_output_file}' has been generated successfully.")
    
   # Save the training metadata to a JSON file
   train_output_file = os.path.join(metadata_dir, "train_metadata.json")
   with open(train_output_file, "w") as f:
       for entry in all_metadata["train"]:
           json.dump(entry, f)
           f.write("\n")

   # Save the ordered test metadata to a JSON file
   test_output_file = os.path.join(metadata_dir, "test_metadata.json")
   with open(test_output_file, "w") as f:
       for entry in ordered_test_metadata:
           json.dump(entry, f)
           f.write("\n")

   print(f"Training metadata file '{train_output_file}' has been generated successfully.")
   print(f"Test metadata file '{test_output_file}' has been generated successfully.")

def generate_all_metadata(ordered_test_metadata, all_metadata):
    """Generates metadata for all unique patches, combining train and test data, 
       and sets clsname to orchard ID for test data in the combined file."""
    all_patches = {}

    # Add training data (no changes)
    for entry in all_metadata["train"]:
        if entry["filename"] not in all_patches:
            all_patches[entry["filename"]] = entry

    # Add test data (set clsname to orchard ID)
    for entry in ordered_test_metadata:
        if entry["filename"] not in all_patches:
            new_entry = entry.copy()  # Create a copy to avoid modifying the original
            orchard_id = entry["filename"].split("\\")[0]  # Extract orchard ID
            new_entry["clsname"] = orchard_id 
            all_patches[entry["filename"]] = new_entry

    all_metadata_list = list(all_patches.values())
    return all_metadata_list

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Generate metadata for orchard dataset")
   parser.add_argument("root_dir", type=str, help="Root directory where your dataset is located")
   parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed statistics")
   parser.add_argument("-t", "--separate", action="store_true", help="Print total separate for each orchard")
   parser.add_argument("-o", "--orchard", action="store_true", help="Instead of creating two seperate train and test metadata files create one file for all patches")
   args = parser.parse_args()

   main(os.path.join(os.path.dirname(os.path.realpath(__file__)), args.root_dir), args.verbose, not(args.separate), args.orchard)
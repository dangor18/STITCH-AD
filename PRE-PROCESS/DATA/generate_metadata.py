import os
import json
import numpy as np
import sys

# set the root directory where your dataset is located
root_dir = sys.argv[1]

# patch labels
class_labels = {
    "normal": 0,
    "case_1": 1,
    "case_2": 2,
    "case_3": 3
}

# Function to process files in a directory
def process_directory(dir_path, class_name, split, orchard_name):
    metadata = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".npy"):
            relative_path = os.path.join(orchard_name, split, class_name, filename)
            metadata_entry = {
                "filename": relative_path,
                "label": class_labels[class_name],
                "clsname": orchard_name,
            }
            metadata.append(metadata_entry)
    return metadata

# Process all orchards
all_metadata = {"train": [], "test": []}

for orchard in os.listdir(root_dir):
    orchard_path = os.path.join(root_dir, orchard)
    if os.path.isdir(orchard_path):
        
        # Process training data
        train_dir = os.path.join(orchard_path, "train")
        if os.path.exists(train_dir):
            normal_dir = os.path.join(train_dir, "normal")
            if os.path.exists(normal_dir):
                all_metadata["train"].extend(process_directory(normal_dir, "normal", "train", orchard))
        
        # Process test data
        test_dir = os.path.join(orchard_path, "test")
        if os.path.exists(test_dir):
            for class_name in class_labels.keys():
                class_dir = os.path.join(test_dir, class_name)
                if os.path.exists(class_dir):
                    all_metadata["test"].extend(process_directory(class_dir, class_name, "test", orchard))

# Create metadata folder if it doesn't exist
metadata_dir = os.path.join(root_dir, "metadata")
if not os.path.exists(metadata_dir):
    os.makedirs(metadata_dir)

# Save the training metadata to a JSON file
train_output_file = os.path.join(metadata_dir, "train_metadata.json")
with open(train_output_file, "w") as f:
    for entry in all_metadata["train"]:
        json.dump(entry, f)
        f.write("\n")

# Save the test metadata to a JSON file
test_output_file = os.path.join(metadata_dir, "test_metadata.json")
with open(test_output_file, "w") as f:
    for entry in all_metadata["test"]:
        json.dump(entry, f)
        f.write("\n")

print(f"Training metadata file '{train_output_file}' has been generated successfully.")
print(f"Test metadata file '{test_output_file}' has been generated successfully.")
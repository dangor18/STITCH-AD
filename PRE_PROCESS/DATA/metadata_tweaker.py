from collections import defaultdict
import argparse
import json
import os
import random

def get_norm_metadata(metadata_path, orchard_list, N):
    """
        ARGS:   metadata_path: path to the training metadata file
                orchard_list: list of orchard ids to filter
        RETURNS: a list of dictionaries containing all training metadata, and a dictionary on orchard specific metadata
    """
    # dictionary to store the removed patches for each orchard
    removed_patches = defaultdict(lambda: 0)
    train_metadata = []
    # load the training metadata file
    with open(metadata_path, "r") as f_r:
        # define dict to a list of dictionaries
        orchard_metas = defaultdict(list)
        count = 0
        for line in f_r:
            nearby = False
            meta = json.loads(line)
            orchard_id = meta["clsname"]
            if orchard_id in orchard_list:
                # filter out normal patches within N pixels of anomalous patches (nearby patches)
                for anom_patch in anom_patch_data[orchard_id]:
                    if abs(meta["x"] - anom_patch["x"]) <= N and abs(meta["y"] - anom_patch["y"]) <= N:
                        #print("Normal patch within N pixels of anomalous patch. Skipping.")
                        removed_patches[orchard_id] += 1
                        nearby = True
                        break
                if not nearby:
                    train_metadata.append(meta)
                    orchard_metas[orchard_id].append({"x": meta["x"], "y": meta["y"], "index": count})
                    count += 1
            else:
                train_metadata.append(meta)
                count += 1
    
    for orchard_id in removed_patches:
        print(f"[INFO] REMOVED {removed_patches[orchard_id]} PATCHES FROM {orchard_id} WITHIN N PIXELS FROM ANOMALOUS PATCHES.")

    return train_metadata, orchard_metas

def get_anom_metadata(metadata_path, orchard_list):
    """
        ARGS:   metadata_path: path to the metadata test file
                orchard_list: list of orchard ids to filter
        RETURNS: a dictionary of anomalous patch data for the orchards
    """
    with open(metadata_path, "r") as f_r:
        anom_patch_data = defaultdict(list)
        for line in f_r:
            meta = json.loads(line)
            # get orchard id in the filename
            orchard_id = os.path.basename(meta["filename"]).split("_")[1]
            if orchard_id in orchard_list and meta["label"] == 1:
                anom_patch_data[orchard_id].append({"x": meta["x"], "y": meta["y"]})
    
    return anom_patch_data

def random_sample(orchard_metas, train_metadata, perc_list, orchard_list):
    """
        ARGS:   orchard_metas: dictionary of orchard specific metadata
                train_metadata: list of dictionaries containing all training metadata
                perc_list: list of percentages to reduce the training data by
                orchard_list: list of orchard ids to reduce
        RETURNS: a new filtered training metadata list, with the specified percentage of patches removed
    """
    result = defaultdict(list)
    # iterate through the orchards to reduce
    for orchard, train_patches in orchard_metas.items():
        # calculate how many patches to keep based on the percentage
        num_patches_to_keep = int(len(train_patches) * (1 - perc_list[orchard_list.index(orchard)] / 100))
        print(f"[INFO] REDUCING {orchard} BY {100 - perc_list[orchard_list.index(orchard)]}% ({num_patches_to_keep} PATCHES REMOVED)")
        # randomly sample the dictionaries from train_patches
        selected_patches = random.sample(train_patches, num_patches_to_keep)
        result[orchard] = selected_patches
    
    # record indices
    indices_to_remove = set()
    for orchard_id in result:
        for patch in result[orchard_id]:
            indices_to_remove.add(patch["index"])
    
    # filter metadata for entire training set, for those indices not in indices_to_remove
    filtered_metadata = [
        entry for idx, entry in enumerate(train_metadata) 
        if idx not in indices_to_remove
    ]

    return filtered_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove / reduce number of training patches from orchards containing artefacts used in evaluation.")
    # take in the path to the metadata folder
    parser.add_argument("-m", "--metadata", type=str, nargs="?", help="path to the metadata file", required=True)
    # list of orchards to reduce and by how much (percentage)
    parser.add_argument("--orchards", type=str, nargs="+", help="list of orchards to reduce and by how much (percentage)", required=True)
    parser.add_argument("-o", "--output", type=str, nargs="?", help="output file name", required=True)
    parser.add_argument("-n", type=int, nargs="?", help="range to consider around anomalous patches to filter from the normal training set", required=False, default=2)
    args = parser.parse_args()
    # use case: python metadata_tweaker.py -o 1676 0 1996 50 2057 12 where the number following the id is the amount to reduce by
    orchard_list = [x for x in args.orchards if args.orchards.index(x) % 2 == 0]
    perc_list = [int(x) for x in args.orchards if args.orchards.index(x) % 2 != 0]

    # get the anomalous patch data
    anom_patch_data = get_anom_metadata(os.path.join(args.metadata, "metadata", "test_metadata.json"), orchard_list)
    
    # full metadata train file
    train_metadata, orchard_metas = get_norm_metadata(os.path.join(args.metadata, "metadata", "train_metadata.json"), orchard_list, args.n)
    
    # filter the training metadata by choosing random samples
    filtered_metadata = random_sample(orchard_metas, train_metadata, perc_list, orchard_list)

    # write the new training metadata file
    with open(os.path.join(args.metadata, "metadata", args.output), "w") as f_w:
        for entry in filtered_metadata:
           json.dump(entry, f_w)
           f_w.write("\n")
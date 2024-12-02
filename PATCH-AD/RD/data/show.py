import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
import random
import cv2 as opencv

case_list = ["NORMAL", "CASE 1", "CASE 2", "CASE 3"]

def plot_image_channels(orchard_data, args):
    """
        Plots the DEM, RGB, and NIR channels of the orchard data. Currently hardcoded to have 5 patches per plot.
        ARGS:
            orchard_data: list of dictionaries containing the metadata of the orchard data
            args: command line arguments
    """
    fig, axs = plt.subplots(5, 4, figsize=(20, 10))
    axs[0][0].set_title('DEM Channel')
    axs[0][0].axis('off')
    axs[0][1].set_title('RGB Image')
    axs[0][1].axis('off')
    axs[0][2].set_title('NIR Channel')
    fig.suptitle("ORCHARD " + orchard_id + f" {str(args.set).upper()} SET", fontsize=16)
    # for each patch in the orchard data
    for i, orchard in enumerate(orchard_data):
        image_path = os.path.join(args.data, orchard["filename"])
        #print(image_path)
        image = np.load(image_path)
        image = opencv.resize(image, (256, 256))
        # get the data channels (rgb, dem, nir)
        rgb_image = image[:, :, 1:4].astype(np.uint8)
        dem = image[:, :, 0]
        nir = image[:, :, 4]

        axs[i][0].imshow(dem, cmap='viridis')
        axs[i][0].axis('off')
        axs[i][1].imshow(rgb_image)
        axs[i][1].axis('off')

        axs[i][2].imshow(nir, cmap='gray')
        axs[i][2].axis('off')

        axs[i][3].axis('off')
        axs[i][3].text(0.5, 0.6, f"PATCH LABEL: {case_list[orchard['label']]}", ha='left', va='center', fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", required=True, help="data path")
    argparser.add_argument("--orchard", required=True, help="orchard id")
    argparser.add_argument("--set", required=True, help="train or test")
    args = argparser.parse_args()

    # create path to metadata file
    if args.set == "train":
        data_path = os.path.join(args.data, "metadata", "train_metadata.json")
    elif args.set == "test":
        data_path = os.path.join(args.data, "metadata", "test_metadata.json")
    else:
        print("Invalid set")
        exit()
    
    # get each line of metadata file
    meta_data = []
    with open(data_path, "r") as f_r:
        for line in f_r:
            meta = json.loads(line)
            meta_data.append(meta)
    
    # limit to only entries with clsname of orchard id
    orchard_id = args.orchard
    orchard_data = [meta for meta in meta_data if meta["clsname"] == orchard_id]
    # shuffle
    random.shuffle(orchard_data)

    # loop through orchard data
    i = 0
    orch_list = []
    for orchard in orchard_data:
        orch_list.append(orchard)
        i += 1
        if i % 5 == 0:
            plot_image_channels(orch_list, args)
            orch_list = []
            cont = input("Continue? (y/n): ")
            if cont == "n":
                break
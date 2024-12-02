import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import yaml
from scipy import ndimage
from sklearn.decomposition import PCA
from skimage import filters
from PIL import Image
from torchvision import transforms
import json
    
class CustomDataset(Dataset):
    def __init__(
        self,
        meta_file,
        data_path,
        transform_fn,
        resize_dim=None,
        noise_factor=0,
        p=0, 
        norm_choice = "IMAGE_NET",
        channels = 3
    ):
        self.meta_file = meta_file
        self.data_path = data_path
        self.transform_fn = transform_fn
        self.resize_dim = resize_dim
        self.noise_factor = noise_factor
        self.p = p
        self.norm_choice = norm_choice
        self.i = 0
        self.channels = channels
        
        # construct metas
        with open(self.meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)
        
        if self.norm_choice == "IMAGE_NET":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #elif self.norm_choice == "PER_ORCHARD" and ("mean" in meta and "std" in meta):
            #self.normalize = transforms.Normalize(mean=np.array(meta["mean"]), std=np.array(meta["std"]))
        else:
            self.normalize = None

    def __len__(self):
        return len(self.metas)
    
    def plot_channels(self, image, title):
        """
            Plot data channels when loading data, used for testing
        """
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title)
        for i, channel_name in enumerate(['dem', 'edge', 'red']):
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
            axs[i].set_title(f'{channel_name} Channel')
            axs[i].axis('off')
            if i == 0:
                plt.colorbar(axs[i].imshow(image[i].numpy(), cmap='gray'), ax=axs[i], label='Value')
            else:
                plt.colorbar(axs[i].imshow(image[i].numpy(), cmap='gray'), ax=axs[i], label='Value')
        plt.tight_layout()
        plt.show()

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]
        
        # read image
        filename = os.path.join(self.data_path, meta["filename"].replace("\\", "/"))
        label = meta["label"]
        image = np.load(filename)
        if self.resize_dim:
            image = cv2.resize(image, self.resize_dim)

        dem = image[:, :, 0]
        dem_min = np.percentile(dem, 5)
        dem_max = np.percentile(dem, 95)
        dem = np.clip((dem - dem_min) / (dem_max - dem_min), 0, 1)
        rgb = image[:, :, 1]
        grey = rgb / 255
        sobel_dem = ndimage.sobel(dem)
        sobel_dem = (sobel_dem - np.min(sobel_dem)) / (np.max(sobel_dem) - np.min(sobel_dem))
        dem = dem[:, :, np.newaxis]
        sobel_dem = sobel_dem[:, :, np.newaxis]
        grey = grey[:, :, np.newaxis]
        image = np.concatenate([dem, sobel_dem, grey], axis=2)
        image = torch.from_numpy(image).float().permute(2, 0, 1)

        input.update(
            {
                "filename": filename,
                "label": label,
            }
        )
        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]
        
        if self.transform_fn:
            image = self.transform_fn(image)

        # normalize
        if self.normalize:
           image = self.normalize(image)

        input.update({"image": image})

        #self.plot_channels(image, "Original Image Channels")

        return input
from random import shuffle
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
    
class inference_dataset(Dataset):
    def __init__(
        self,
        meta_file_1,
        meta_file_2,
        data_path,
        resize_dim=None,
    ):
        self.meta_file_1 = meta_file_1
        self.meta_file_2 = meta_file_2
        self.data_path = data_path
        self.resize_dim = resize_dim
        
        # construct metas
        with open(meta_file_1, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)
        
        with open(meta_file_2, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)
        
        # NOTE: be careful of this
        self.metas = shuffle(self.metas)    # shuffle the meta data to properly mix in the train and test set (get all patches then in one loader)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]
        
        # read image
        filename = os.path.join(self.data_path, meta["filename"].replace("\\", "/"))
        label = meta["label"]
        image = np.load(filename)
        if self.resize_dim:
            image = cv2.resize(image, self.resize_dim)

        # only DEM
        if np.ndim(image) == 2:
            image = torch.unsqueeze(torch.from_numpy(image).float(), dim=0)
            self.normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        else:
            dem = image[:, :, 0]
            #dem = torch.unsqueeze(torch.from_numpy(dem).float(), dim=0)
            rgb = image[:, :, 1]
            grey = rgb
            #grey = torch.unsqueeze(torch.from_numpy(grey).float(), dim=0)
            #prewitt_dem = filters.prewitt(dem)
            sobel_dem = ndimage.sobel(dem)

            #prewitt_dem = prewitt_dem[:, :, np.newaxis]
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

        return input
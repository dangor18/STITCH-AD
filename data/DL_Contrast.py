import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy import ndimage
from sklearn.decomposition import PCA
from skimage import filters
from data.noise import Simplex_CLASS
from torchvision import transforms
import json
import random
    
class train_dataset(Dataset):
    def __init__(
        self,
        meta_file,
        data_path,
        resize_dim=(256, 256),
    ):
        self.meta_file = meta_file
        self.data_path = data_path
        self.resize_dim = resize_dim
        self.simplexNoise = Simplex_CLASS()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #self.random_erasing = transforms.RandomErasing(p=0.25, scale=(0.5, 0.8), ratio=(0.3, 3.3), value=0, inplace=False)
        
        # construct metas
        with open(self.meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)
    
    def plot_channels(self, image, title):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, size=16)
        for i, channel_name in enumerate(['DEM', 'Edge', 'Red']):
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
            axs[i].set_title(f'{channel_name} Channel')
            axs[i].axis('off')
            if i == 0:
                plt.colorbar(axs[i].imshow(image[i].numpy(), cmap='viridis'), ax=axs[i], label='Value')
            else:
                plt.colorbar(axs[i].imshow(image[i].numpy(), cmap='gray'), ax=axs[i], label='Value')
        plt.tight_layout()
        plt.show()

    def get_psuedo_case1(self, dem, seed=None):
        if seed is not None:
            np.random.seed(seed)
    
        size = 256
        h_noise = np.random.randint(128, 200)
        w_noise = np.random.randint(128, 200)
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        
        # Random scales and rotation
        scale_x = np.random.uniform(0.5, 2.0)
        scale_y = np.random.uniform(0.5, 2.0)
        rotation = np.random.uniform(0, 2*np.pi)
    
        y, x = np.ogrid[:h_noise, :w_noise]
        x = x / w_noise - 0.5
        y = y / h_noise - 0.5
    
        # Apply rotation
        x_rot = x * np.cos(rotation) - y * np.sin(rotation)
        y_rot = x * np.sin(rotation) + y * np.cos(rotation)
    
        gradient_x = x_rot * scale_x
        gradient_y = y_rot * scale_y
    
        gradient = gradient_x + gradient_y
        gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient))

        mean = np.mean(dem[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise])
        std = np.std(dem[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise])
        # Random intensity
        intensity = np.random.uniform(3, 4) * std
        gradient = gradient * intensity + mean - intensity/2
        
        # Apply the gradient to the specified region
        dem_copy = dem.copy()
        dem_copy[start_h_noise:start_h_noise + h_noise, start_w_noise:start_w_noise + w_noise] = gradient
        
        return dem_copy

    def get_psuedo_case2(self, dem, amplitude=1):
        # add simplex noise to create pseudo abnormal sample
        size = 256
        h_noise = np.random.randint(128, 200)
        w_noise = np.random.randint(128, 200)
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 9, 0.8)
        init_noise = np.zeros((256, 256, 3))
        std = np.std(dem[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise])
        init_noise[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = simplex_noise.transpose(1,2,0) * std * amplitude
        dem_noise = dem + init_noise[:, :, 0]
        return dem_noise
    
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
            normal_image = torch.unsqueeze(torch.from_numpy(image).float(), dim=0)
            noise = self.get_noise()
            noise *= np.std(image)  # change std dev to match that of orchard, as without this the noise affects some orchard patches more than others
            img_noise = image + noise
            #img_noise = self.overlay_blend(image, noise)
            img_noise = torch.unsqueeze(torch.from_numpy(img_noise).float(), dim=0)
            img_noise.clamp(0, 1)
            self.normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        else:
            dem = image[:, :, 0]
            dem_min = np.percentile(dem, 5)
            dem_max = np.percentile(dem, 95)
            dem = np.clip((dem - dem_min) / (dem_max - dem_min), 0, 1)
            grey = image[:, :, 1] / 255
            sobel_dem = ndimage.sobel(dem)
            sobel_dem = (sobel_dem - sobel_dem.min()) / (sobel_dem.max() - sobel_dem.min())

            dem_na= dem[:, :, np.newaxis]
            sobel_dem_na = sobel_dem[:, :, np.newaxis]
            grey_na = grey[:, :, np.newaxis]
            normal_image = np.concatenate([dem_na, sobel_dem_na, grey_na], axis=2)
            normal_image = torch.from_numpy(normal_image).float().permute(2, 0, 1)
            
            choice = random.choice([1, 2])
            if choice == 1:
                dem_noise = self.get_psuedo_case1(dem)
            elif choice == 2:
                dem_noise = self.get_psuedo_case2(dem, amplitude=0.7)

            sobel_noise = ndimage.sobel(dem_noise)
            sobel_noise = (sobel_noise - sobel_noise.min()) / (sobel_noise.max() - sobel_noise.min())
            dem_na = dem_noise[:, :, np.newaxis]
            grey_na = grey[:, :, np.newaxis]
            sobel_noise = sobel_noise[:, :, np.newaxis]

            img_noise = np.concatenate([dem_na, sobel_noise, grey_na], axis=2)
            img_noise = torch.from_numpy(img_noise).float().permute(2, 0, 1)
        
        if self.normalize:
            img_noise = self.normalize(img_noise)
        # normalize NORMAL image patch
        if self.normalize:
            normal_image = self.normalize(normal_image)

        input.update(
            {
                "filename": filename,
                "label": label,
                "normal_image": normal_image,
                "abnormal_image": img_noise,
            }
        )
        if meta.get("clsname", None):
            input["clsname"] = meta["clsname"]
        else:
            input["clsname"] = filename.split("/")[-4]

        #print(normal_image)
        #print(img_noise)
        #self.plot_channels(img_noise, "Psuedo Stitching Artefact Channels")
        #self.plot_channels(normal_image, "Normal Image Channels")

        return input
    
class test_dataset(Dataset):
    def __init__(
        self,
        meta_file,
        data_path,
        resize_dim=(256, 256),
    ):
        self.meta_file = meta_file
        self.data_path = data_path
        self.resize_dim = resize_dim
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)
    
    def plot_channels(self, image, title):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, size=16)
        for i, channel_name in enumerate(['DEM', 'Edge', 'Red']):
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
            axs[i].set_title(f'{channel_name} Channel')
            axs[i].axis('off')
            if i == 0:
                plt.colorbar(axs[i].imshow(image[i].numpy(), cmap='viridis'), ax=axs[i], label='Value')
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

        # only DEM
        if np.ndim(image) == 2:
            image = torch.unsqueeze(torch.from_numpy(image).float(), dim=0)
            self.normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        else:
            dem = image[:, :, 0]
            dem_min = np.percentile(dem, 5)
            dem_max = np.percentile(dem, 95)
            dem = np.clip((dem - dem_min) / (dem_max - dem_min), 0, 1)
            rgb = image[:, :, 1]
            grey = rgb / 255
            sobel_dem = ndimage.sobel(dem)
            sobel_dem = (sobel_dem - sobel_dem.min()) / (sobel_dem.max() - sobel_dem.min())

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

        # normalize
        if self.normalize:
            image = self.normalize(image)

        input.update({"image": image})

        #self.plot_channels(image, "Original Image Channels")
        #self.plot_channels(noisy_image, "Noisy Image Channels")

        return input
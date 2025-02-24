import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy import ndimage
from data.noise import Simplex_CLASS
from torchvision import transforms
import json
import random

from dataclasses import dataclass
from typing import Tuple

@dataclass
class AugmentationParams:
    p_flip: float = 0.5
    p_rotate: float = 0.5
    p_crop: float = 0.5
    p_blur: float = 0.1
    p_noise: float = 0.1

@dataclass
class SimplexNoiseParams:
    octaves: int = 9
    persistence: float = 0.8
    amplitude: float = 0.7
    frequency: int = 32

def plot_channels(image, title):
    """
        Plot data channels when loading data, used for testing
    """
    channel_names = ['DEM', 'Edge', 'Red', 'Red spec', 'Red Edge', 'nir']
    # number of plots for each channel
    n = image.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(15, 5))
    fig.suptitle(title, size=20)
    for i in range(n):
        axs[i].set_xlabel('X', size=16)
        axs[i].set_ylabel('Y', size=16)
        axs[i].set_title(f'{channel_names[i]} Channel', size=18)
        axs[i].axis('off')
        if i == 0:
            plt.colorbar(axs[i].imshow(image[i].numpy(), cmap='viridis'), ax=axs[i], label='Value')
        else:
            plt.colorbar(axs[i].imshow(image[i].numpy(), cmap='gray'), ax=axs[i], label='Value')
    plt.tight_layout()
    plt.show()
    
class train_dataset(Dataset):
    def __init__(
        self,
        meta_file,
        data_path,
        resize_dim=(256, 256),
        transform_fn=False,
        aug_params: AugmentationParams = AugmentationParams(),
        simplex_params: SimplexNoiseParams = SimplexNoiseParams(),
        in_channels=3,
    ):
        self.meta_file = meta_file
        self.data_path = data_path
        self.resize_dim = resize_dim
        self.simplexNoise = Simplex_CLASS()

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.in_channels = in_channels

        if self.in_channels > 3:
            # repeat above norms and std for each in channel
            imagenet_mean = imagenet_mean * (self.in_channels // 3) + imagenet_mean[:self.in_channels % 3]
            imagenet_std = imagenet_std * (self.in_channels // 3) + imagenet_std[:self.in_channels % 3]

        self.normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        self.transform_fn = transform_fn
        self.aug_params = aug_params
        self.simplex_params = simplex_params
        
        # construct metas
        with open(self.meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def get_psuedo_case1(self, dem, seed=None):
        """
            Returns psuedo case 1 by erasing a random portion of the dem and applying a smoothed out gradient
        """
        if seed is not None:
            np.random.seed(seed)

        # set sizes
        size = 256
        h_noise = np.random.randint(128, 200)
        w_noise = np.random.randint(128, 200)
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        
        # random scales and rotation
        scale_x = np.random.uniform(0.5, 2.0)
        scale_y = np.random.uniform(0.5, 2.0)
        rotation = np.random.uniform(0, 2*np.pi)
    
        y, x = np.ogrid[:h_noise, :w_noise]
        x = x / w_noise - 0.5
        y = y / h_noise - 0.5
    
        # apply rotation
        x_rot = x * np.cos(rotation) - y * np.sin(rotation)
        y_rot = x * np.sin(rotation) + y * np.cos(rotation)
    
        gradient_x = x_rot * scale_x
        gradient_y = y_rot * scale_y
    
        gradient = gradient_x + gradient_y
        gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient))

        # get mean and std dev of region you're erasing to normalize grad
        mean = np.mean(dem[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise])
        std = np.std(dem[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise])

        intensity = np.random.uniform(3, 4) * std
        gradient = gradient * intensity + mean - intensity/2
        
        # apply the gradient to the specified region
        dem_copy = dem.copy()
        dem_copy[start_h_noise:start_h_noise + h_noise, start_w_noise:start_w_noise + w_noise] = gradient
        
        return dem_copy

    def get_psuedo_case2(self, dem):
        """"
            Return psuedo case 2 by adding simplex noise to the dem
        """
        # add simplex noise to create pseudo abnormal sample
        size = 256
        h_noise = np.random.randint(128, 200)
        w_noise = np.random.randint(128, 200)
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), self.simplex_params.octaves, self.simplex_params.persistence, self.simplex_params.frequency)
        init_noise = np.zeros((256, 256, 3))
        std = np.std(dem[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise])
        init_noise[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = simplex_noise.transpose(1,2,0) * std * self.simplex_params.amplitude
        dem_noise = dem + init_noise[:, :, 0]
        return dem_noise
    
    def add_gaussian_noise(self, image, mean=0., std=0.1):
        noise = torch.randn_like(image) * std + mean

        return image + noise

    def apply_variational_augs(self, DEM):
        """
            Apply slight gaussian blur and random noise to the DEM
        """
        # make tensor
        DEM = torch.from_numpy(DEM).float().unsqueeze(0)
        # apply gaussian blur
        if random.random() < self.aug_params.p_blur:
            DEM = F.gaussian_blur(DEM, kernel_size=(5, 5))
        # apply random noise
        if random.random() < self.aug_params.p_noise:
            DEM = self.add_gaussian_noise(DEM)
        # return as numpy array
        return DEM.squeeze(0).numpy()
 
    def apply_geometric_augs(self, normal_img, noise_img):
        """
            Apply random flips to the normal and noise image
        """
        if random.random() < self.aug_params.p_flip:
            normal_img = F.hflip(normal_img)
            noise_img = F.hflip(noise_img)
        if random.random() < self.aug_params.p_flip:
            normal_img = F.vflip(normal_img)
            noise_img = F.vflip(noise_img)
        if random.random() < self.aug_params.p_rotate:
            angle = random.randint(0, 360)
            normal_img = F.rotate(normal_img, angle)
            noise_img = F.rotate(noise_img, angle)
        if random.random() < self.aug_params.p_crop:
            scale = random.uniform(1.0, 1.3)
            normal_img = F.affine(normal_img, angle=0, translate=[0, 0], scale=scale, shear=0)
            noise_img = F.affine(noise_img, angle=0, translate=[0, 0], scale=scale, shear=0)
        
        return normal_img, noise_img
    
    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]
        
        # read image
        filename = os.path.join(self.data_path, meta["filename"].replace("\\", "/"))
        image = np.load(filename)
        if self.resize_dim:
            image = cv2.resize(image, self.resize_dim)

        # if image only contains dem and therefore 2D
        image = image[:, :, np.newaxis] if len(image.shape) == 2 else image

        # get dem, scale and create sobel dem
        dem = image[:, :, 0]
        # add small amounts of 'natural noise' to dem (blur and noise)
        if self.transform_fn:
            dem = self.apply_variational_augs(dem)

        dem_min = np.percentile(dem, 1)
        dem_max = np.percentile(dem, 99)
        # scale dem in image
        dem = np.clip((dem - dem_min) / (dem_max - dem_min), 0, 1)
        image[:, :, 0] = dem
        # insert sobel after dem in image
        sobel_dem = ndimage.sobel(dem)
        sobel_dem = (sobel_dem - sobel_dem.min()) / (sobel_dem.max() - sobel_dem.min())
        image = np.concatenate([image[:, :, 0:1], sobel_dem[:, :, np.newaxis], image[:, :, 1:]], axis=2)

        if self.in_channels >= 3:
            # Process RGB
            # remove the green and blue channels from the RGB (4rd and 5th channels in image)
            image = np.concatenate((image[:, :, 0:3], image[:, :, 5:]), axis=2)
            image[:, :, 2] = image[:, :, 2] / 255

        # Process spectral channels
        for i in range(3, self.in_channels):
            image[:, :, i] = (image[:, :, i] - meta["min_vals"][i+1]) / (meta["max_vals"][i+1] - meta["min_vals"][i+1])

        # delete remaining channels (after > in_channels)
        image = image[:, :, :self.in_channels]
        normal_image = torch.from_numpy(image).float().permute(2, 0, 1)
        
        # randomly choose either case 1 or 2 psuedo-artefact
        choice = random.choice([1, 2])
        if choice == 1:
            dem_noise = self.get_psuedo_case1(dem)
        elif choice == 2:
            dem_noise = self.get_psuedo_case2(dem)

        image_noise = image.copy()
        sobel_noise = ndimage.sobel(dem_noise)
        sobel_noise = (sobel_noise - sobel_noise.min()) / (sobel_noise.max() - sobel_noise.min())
        image_noise[:, :, 0] = dem_noise
        image_noise[:, :, 1] = sobel_noise
        
        img_noise = torch.from_numpy(image_noise).float().permute(2, 0, 1)

        # repeat DEM
        if self.in_channels == 2:
            normal_image = torch.concat([normal_image, normal_image[0].unsqueeze(0)], dim=0)
            img_noise = torch.concat([img_noise, img_noise[0].unsqueeze(0)], dim=0)
        
        # apply flips
        if self.transform_fn:
            normal_image, img_noise = self.apply_geometric_augs(normal_image, img_noise)

        # normalize
        if self.normalize:
            img_noise = self.normalize(img_noise)
            normal_image = self.normalize(normal_image)

        input.update(
            {
                "filename": filename,
                "label": meta["label"],
                "case": meta["case"],
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
        #plot_channels(normal_image, "Normal Image Channels")
        #plot_channels(img_noise, "Psuedo Stitching Artefact Channels")

        return input
    
class test_dataset(Dataset):
    def __init__(
        self,
        meta_file,
        data_path,
        resize_dim=(256, 256),
        in_channels=3,
    ):
        self.meta_file = meta_file
        self.data_path = data_path
        self.resize_dim = resize_dim
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.in_channels = in_channels

        if self.in_channels > 3:
            # repeat above norms and std for each in channel
            imagenet_mean = imagenet_mean * (self.in_channels // 3) + imagenet_mean[:self.in_channels % 3]
            imagenet_std = imagenet_std * (self.in_channels // 3) + imagenet_std[:self.in_channels % 3]

        self.normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]
        
        # read image
        filename = os.path.join(self.data_path, meta["filename"].replace("\\", "/"))
        image = np.load(filename)
        if self.resize_dim:
            image = cv2.resize(image, self.resize_dim)

        # if image only contains dem and therefore 2D
        image = image[:, :, np.newaxis] if len(image.shape) == 2 else image
        
        # get dem, scale and create sobel dem
        dem = image[:, :, 0]
        dem_min = np.percentile(dem, 1)
        dem_max = np.percentile(dem, 99)
        # scale dem in image
        image[:, :, 0] = np.clip((dem - dem_min) / (dem_max - dem_min), 0, 1)
        # insert sobel after dem in image
        sobel_dem = ndimage.sobel(dem)
        sobel_dem = (sobel_dem - sobel_dem.min()) / (sobel_dem.max() - sobel_dem.min())
        image = np.concatenate([image[:, :, 0:1], sobel_dem[:, :, np.newaxis], image[:, :, 1:]], axis=2)

        if self.in_channels >= 3:
            # Process RGB
            # remove the green and blue channels from the RGB (4rd and 5th channels in image)
            image = np.concatenate((image[:, :, 0:3], image[:, :, 5:]), axis=2)
            image[:, :, 2] = image[:, :, 2] / 255

        for i in range(3, self.in_channels):
            image[:, :, i] = (image[:, :, i] - meta["min_vals"][i+1]) / (meta["max_vals"][i+1] - meta["min_vals"][i+1])

        # delete remaining channels (after > in_channels)
        image = image[:, :, :self.in_channels]
        image = torch.from_numpy(image).float().permute(2, 0, 1)

        # repeat DEM
        if self.in_channels == 2:
            image = torch.concat([image, image[0].unsqueeze(0)], dim=0)

        input.update(
            {
                "filename": filename,
                "label": meta["label"],
                "case": meta["case"],
                "x": meta["x"],
                "y": meta["y"],
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

        #plot_channels(image, "Artefact Image Channels")
        #self.plot_channels(noisy_image, "Noisy Image Channels")

        return input
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

# TODO: REMOVE THIS IF NOT USED
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
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
        self.random_erasing = transforms.RandomErasing(p=0.05, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        
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
        fig.suptitle(title)
        for i, channel_name in enumerate(['dem', 'nir', 'reg']):
            axs[i].imshow(image[i].numpy(), cmap='gray')
            axs[i].set_title(f'{channel_name} Channel')
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()

    def overlay_blend(self, base, top):
        return np.where(base > 0.5, 1 - 2*(1-base)*(1-top), 2*base*top)

    def get_noise(self):
        # add simplex noise to create pseudo abnormal sample
        size = 256
        h_noise = np.random.randint(128, 200)
        w_noise = np.random.randint(128, 200)
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 8, 0.6)
        
        # Normalize simplex noise to range [0, 1]
        simplex_noise = (simplex_noise - simplex_noise.min()) / (simplex_noise.max() - simplex_noise.min()) * 0.5
        simplex_noise = (simplex_noise - simplex_noise.mean()) / simplex_noise.std()
        
        # Initialize with 0.5 (neutral gray) instead of 0
        init_noise = np.zeros((256, 256, 3))
        init_noise[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = simplex_noise.transpose(1,2,0)

        return init_noise[:, :, 0]
    
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
            img_noise = self.random_erasing(img_noise)
            img_noise.clamp(0, 1)
            self.normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        else:
            dem = image[:, :, 0]
            grey = image[:, :, 1]
            sobel_dem = filters.prewitt(dem)
            dem_na= dem[:, :, np.newaxis]
            sobel_dem_na = sobel_dem[:, :, np.newaxis]
            grey_na = grey[:, :, np.newaxis]
            normal_image = np.concatenate([dem_na, sobel_dem_na, grey_na], axis=2)
            normal_image = torch.from_numpy(normal_image).float().permute(2, 0, 1)

            noise = self.get_noise()
            dem_noise = noise * np.std(dem)  # change std dev to match that of orchard, as without this the noise affects some orchard patches more than others
            dem_noise = dem + dem_noise
            grey_noise = noise * np.std(grey)
            grey_noise = grey + grey_noise
            grey_na = grey_noise[:, :, np.newaxis]
            sobel_noise = filters.prewitt(dem_noise)
            dem_na = dem_noise[:, :, np.newaxis]
            sobel_noise = sobel_noise[:, :, np.newaxis]
            img_noise = np.concatenate([dem_na, sobel_noise, grey_na], axis=2)
            img_noise = torch.from_numpy(img_noise).float().permute(2, 0, 1)
            img_noise = self.random_erasing(img_noise)
            img_noise.clamp(0, 1)
            #print(simplex_noise.min(), simplex_noise.max())
        
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

        print(normal_image)
        print(img_noise)
        self.plot_channels(normal_image, "Original Image Channels")
        self.plot_channels(img_noise, "Noisy Image Channels")

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
        fig.suptitle(title)
        for i, channel_name in enumerate(['dem', 'nir', 'reg']):
            axs[i].imshow(image[i].numpy(), cmap='gray')
            axs[i].set_title(f'{channel_name} Channel')
            axs[i].axis('off')
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
            #dem = torch.unsqueeze(torch.from_numpy(dem).float(), dim=0)
            rgb = image[:, :, 1]
            '''h, w, c = rgb.shape
            rgb_reshaped = rgb.reshape(-1, c)
            
            # Apply PCA
            pca = PCA(n_components=1)
            grey = pca.fit_transform(rgb_reshaped)
            
            # Reshape grey back to 2D and convert to PyTorch tensor
            grey = grey.reshape(h, w)'''
            grey = rgb
            #grey = torch.unsqueeze(torch.from_numpy(grey).float(), dim=0)
            #prewitt_dem = filters.prewitt(dem)
            sobel_dem = filters.sobel(dem)

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

        # normalize
        if self.normalize:
            image = self.normalize(image)

        input.update({"image": image})

        #self.plot_channels(image, "Original Image Channels")
        #self.plot_channels(noisy_image, "Noisy Image Channels")

        return input
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import yaml
from scipy import ndimage
from sklearn import decomposition
from skimage import filters
from PIL import Image
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
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)
        
        if self.norm_choice == "IMAGE_NET":
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif self.norm_choice == "PER_ORCHARD" and ("mean" in meta and "std" in meta):
            self.normalize = transforms.Normalize(mean=np.array(meta["mean"]), std=np.array(meta["std"]))
        else:
            self.normalize = None

    def __len__(self):
        return len(self.metas)
    # TODO: REMOVE THESE IF NOT USED
    def calculate_ndvi(self, image_array):
        """
        Calculate NDVI from a NumPy array with shape (H, W, C) where C=3,
        and channel 2 is NIR, channel 3 is RED.
        
        Args:
        image_array (numpy.ndarray): Input image array with shape (H, W, 3)
        
        Returns:
        numpy.ndarray: NDVI array with shape (H, W)
        """
        # extract NIR and RED spectral channels
        nir = image_array[:, :, 1].astype(np.float32)
        red = image_array[:, :, 2].astype(np.float32)
        
        numerator = nir - red
        denominator = nir + red
        
        ndvi = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        ndvi = np.clip(ndvi, 0, 1)
        return ndvi

    def detect_dem_artifacts(self, image_array, smooth_sigma=1, diff_threshold=0.2):
        """
        Detect potential artifacts in DEM by comparing with NDVI.
        
        Args:
        image_array (numpy.ndarray): Input image array with shape (H, W, C)
        dem_index (int): Index of DEM channel (default: 0)
        nir_index (int): Index of NIR channel (default: 1)
        red_index (int): Index of Red channel (default: 2)
        smooth_sigma (float): Sigma for Gaussian smoothing (default: 1)
        diff_threshold (float): Threshold for considering a difference significant (default: 0.2)
        
        Returns:
        numpy.ndarray: Array highlighting potential artifacts
        """
        #ndvi = self.calculate_ndvi(image_array)
        #dem = filters.prewitt(image_array[:, :, 0].astype(np.float32))
        # Normalize DEM and NDVI to 0-1 range
        #ndvi_norm = (ndvi - np.min(ndvi)) / (np.max(ndvi) - np.min(ndvi))
        
        # Calculate difference
        #diff = np.abs(dem - ndvi)
        #diff = np.clip(diff, 0, 1)
        # Apply Gaussian smoothing to reduce noise
        #diff_smooth = ndimage.gaussian_filter(diff, sigma=smooth_sigma)
        #thresh = np.mean(diff_smooth) + diff_threshold * np.std(diff_smooth)
        # Threshold the difference to highlight potential artifacts
        #artifacts = np.where(diff_smooth > thresh, 1, 0)
        #temp = self.threshold_and_xor(dem, ndvi)
        #return dem
    
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
            rgb = image[:, :, 1:]
            grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
            #prewitt_dem = filters.prewitt(dem)
            sobel_dem = ndimage.sobel(dem)

            #prewitt_dem = prewitt_dem[:, :, np.newaxis]
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

        #noisy_image = self.add_noise(image, self.noise_factor, self.p)
        #input.update({"noisy_image": noisy_image})
        #self.plot_channels(image, "Original Image Channels")
        #self.plot_channels(noisy_image, "Noisy Image Channels")

        return input

    @staticmethod
    def add_noise(image, noise_factor, p):
        '''
        Function to randomly add noise to the image.
        Parameters:
        ----------
        img : torch.Tensor
            The image to add noise to.
        noise_factor : float
            The factor to multiply the noise by.
        p : float
            The probability of adding noise to the image.
        '''
        if np.random.rand() > p:
            return image
        noise = torch.randn_like(image) * noise_factor
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0., 1.)
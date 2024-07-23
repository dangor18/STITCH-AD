import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import yaml
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
        norm_choice = "IMAGE_NET"
    ):
        self.meta_file = meta_file
        self.data_path = data_path
        self.transform_fn = transform_fn
        self.resize_dim = resize_dim
        self.noise_factor = noise_factor
        self.p = p
        self.norm_choice = norm_choice
        self.i = 0
        
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
            self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.metas)

    def plot_channels(self, image, title):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title)
        for i, channel_name in enumerate(['Red']):
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

        #print(np.shape(image))
        #image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0    # equivalent to ToTensor just without using uint8
        if np.ndim(image) == 2:
            image = torch.unsqueeze(torch.from_numpy(image).float(), dim=0)
            self.normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1)
        #image = torch.squeeze(image)
        #image = transforms.ToTensor()(image)

        #image = Image.fromarray(image, mode='RGB')
        #image = transforms.ToTensor()(image)
        #print(image)

        if self.transform_fn:
            image = self.transform_fn(image)

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
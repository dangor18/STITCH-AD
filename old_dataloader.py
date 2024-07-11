import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import json
from torchvision import transforms
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(
        self,
        meta_file,
        data_path,
        transform_fn,
        resize_dim=None,
        noise_factor=0,
        p=0
    ):
        self.meta_file = meta_file
        self.data_path = data_path
        self.transform_fn = transform_fn
        self.resize_dim = resize_dim
        self.noise_factor = noise_factor
        self.p = p
       
        # Load meta information
        with open(meta_file, "r") as f_r:
            self.metas = [json.loads(line) for line in f_r]
       
        # Prepare normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    def add_noise(image, noise_factor, p):
        if np.random.rand() > p:
            return image
        noise = torch.randn_like(image) * noise_factor
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0., 1.)

    def load_and_process_image(self, filename):
        image = np.load(filename)
        if self.resize_dim:
            image = cv2.resize(image, self.resize_dim)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        if self.transform_fn:
            image = self.transform_fn(image)
        image = self.normalize(image)
        return image

    def __len__(self):
        return len(self.metas)

    def plot_channels(self, image, title):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title)
        for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
            axs[i].imshow(image[i].numpy(), cmap='gray')
            axs[i].set_title(f'{channel_name} Channel')
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()

    def __getitem__(self, index):
        meta = self.metas[index]
        filename = os.path.join(self.data_path, meta["filename"])
        label = meta["label"]
        image = self.load_and_process_image(filename)
        noisy_image = self.add_noise(image, self.noise_factor, self.p)
        
        # Plot each channel separately in grayscale
        #self.plot_channels(image, "Original Image Channels")
        #self.plot_channels(noisy_image, "Noisy Image Channels")
        print(image)

        return {
            "filename": filename,
            "label": label,
            "clsname": meta.get("clsname", os.path.basename(os.path.dirname(filename))),
            "image": image,
            "noisy_image": noisy_image
        }
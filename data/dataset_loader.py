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
        p=0
    ):
        self.meta_file = meta_file
        self.data_path = data_path
        self.transform_fn = transform_fn
        self.resize_dim = resize_dim
        self.noise_factor = noise_factor
        self.p = p
        self.i= 0

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        # read image
        filename = os.path.join(self.data_path, meta["filename"])
        label = meta["label"]

        image = np.load(filename)

        if self.resize_dim:
            image = cv2.resize(image, self.resize_dim)
        
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

        image = Image.fromarray(image.squeeze(), mode="RGB")

        if self.transform_fn:
            image = self.transform_fn(image)

        image = transforms.ToTensor()(image)
        
        # cut image down to one channel
        #image = image[0].unsqueeze(0)

        if "mean" in meta and "std" in meta:
            #mean=np.array(meta["mean"])/255
            #std=np.array(meta["std"])/255
            image = self.normalize(image)
        #if torch.any(image > 1) or torch.any(image < -1):
        #    print(f"Warning: Image {filename} has values outside [-1, 1] range.")
        #    print(f"Min value: {image.min().item()}, Max value: {image.max().item()}")
        input.update({"image": image})
        noisy_image = self.add_noise(image, self.noise_factor, self.p)
        input.update({"noisy_image": noisy_image})
        
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
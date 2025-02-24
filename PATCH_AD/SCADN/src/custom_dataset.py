import glob
import random
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
from torch import from_numpy
from sklearn.model_selection import train_test_split
from scipy import ndimage

class StitchoDataset(Dataset):
    def __init__(
        self,
        meta_file,
        transform_fn,
        resize_dim=None, 
        noise_factor=0, 
        p=0,
        dataroot=None
    ):
        self.meta_file = meta_file
        self.transform_fn = transform_fn
        self.resize_dim = resize_dim
        self.noise_factor = noise_factor
        self.p = p
        self.dataroot = dataroot
        self.class_count = {}

        # construct metas
        with open(meta_file, "r") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

                # keep track of class count
                if meta["clsname"] not in self.class_count:
                    self.class_count[meta["clsname"]] = 0
                self.class_count[meta["clsname"]] += 1


    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        input = {}
        meta = self.metas[index]

        filename = meta["filename"].replace("\\", "/")

        # read image
        filename = os.path.join(self.dataroot, filename)
        label = meta["label"]
        if os.path.exists(filename):
            image = np.load(filename)
        else:
            print(f"File {filename} does not exist.")
            exit()

        # print(image.shape)
        # exit()

        if self.resize_dim:
            image = cv2.resize(image, self.resize_dim)

        # add extra sobel-filtered layer
        sobel = ndimage.sobel(image[:, :, 0])
        sobel = np.expand_dims(sobel, axis=2)
        image = np.concatenate((image, sobel), axis=2)

        # scale layers to 0-1
        for i in range(image.shape[2]):
            min = np.min(image[:, :, i])
            max = np.max(image[:, :, i])
            image[:, :, i] = (image[:, :, i] - min) / (max - min)

        # # use numpy to show each layer in the image
        # num_images = image.shape[2]
        # fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        # for i in range(num_images):
        #     print(f'layer {i}: {np.min(image[:, :, i])} - {np.max(image[:, :, i])}')
        #     axes[i].imshow(image[:, :, i])
        #     axes[i].axis('off')
        # plt.show()
        # exit()
        
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

        # print(image.shape)
        # exit()

        if len(image.shape) == 3:
            image = from_numpy(image).float().permute(2, 0, 1)
        else:
            image = from_numpy(image).float()[None, :, :]

        # print(image.shape)
        # exit()

        if self.transform_fn:
            image = self.transform_fn(image)
        
        # if "mean" in meta and "std" in meta:
        #     normalize_fn = transforms.Normalize(mean=meta["mean"][0], std=meta["std"][0])
        #     # print(image.shape)
        #     image = normalize_fn(image)
            
            
        # # duplicate channels of image to 3
        # if image.size(0) == 1:
        #     image = image.expand(3, -1, -1)
            
        input.update({"image": image})
        # noisy_image = add_noise(image, self.noise_factor, self.p)
        # input.update({"noisy_image": noisy_image})
        
        return input['image'], input['label'], input['clsname'], meta['x'], meta['y']
    
    def get_class_count(self):
        return {clsname: count for clsname, count in self.class_count.items()}
    
def load_data(opt):

    

    train_metadata = os.path.join(opt.dataroot, 'metadata', opt.train_meta)
    test_metadata = os.path.join(opt.dataroot, 'metadata', opt.test_meta)

    splits = ['train', 'test', 'train4val']

    masks = AddMask(opt)
    collate = {'train': masks.append_mask, 'test': masks.append_mask, 'train4val': masks.append_mask} # collate functions for dataloader

    splits2metadata = {'train': train_metadata, 'test': test_metadata, 'train4val': train_metadata}
    drop_last_batch = {'train': True, 'test': False, 'train4val': False}
    shuffle = {'train': True, 'test': False, 'train4val': False}

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    num_channels = opt.INPUT_CHANNELS

    means = imagenet_mean * (num_channels // 3) + imagenet_mean[:num_channels % 3]
    stds = imagenet_std * (num_channels // 3) + imagenet_std[:num_channels % 3]

    if opt.MODIFIED:
        print('modified')
        transform = transforms.Compose([transforms.Resize(opt.INPUT_SIZE),
                                        transforms.CenterCrop(opt.INPUT_SIZE),
                                        transforms.RandomRotation(degrees=45),
                                        transforms.Normalize(mean=means, std=stds), ])
    else:
        transform = transforms.Compose([transforms.Resize(opt.INPUT_SIZE),
                                        transforms.CenterCrop(opt.INPUT_SIZE),
                                        transforms.Normalize(mean=means, std=stds), ])

    dataset = {x: StitchoDataset(meta_file=splits2metadata[x], transform_fn=transform, resize_dim=(opt.INPUT_SIZE, opt.INPUT_SIZE), dataroot=opt.dataroot) for x in splits}

    # split train dataset into train and validation datasets
    train_idx, val_idx = train_test_split(list(range(len(dataset['train']))), test_size=opt.VAL_SPLIT, random_state=opt.SEED)
    dataset['train4val'] = torch.utils.data.Subset(dataset['train'], val_idx)
    dataset['train'] = torch.utils.data.Subset(dataset['train'], train_idx)

    dataloader = {}

    for x in splits:
        if collate[x] is not None:
            dataloader[x] = torch.utils.data.DataLoader(dataset=dataset[x],
                                                        batch_size=opt.BATCH_SIZE,
                                                        shuffle=shuffle[x],
                                                        num_workers=int(opt.workers),
                                                        drop_last=drop_last_batch[x],
                                                        collate_fn=collate[x])
        else:
            dataloader[x] = torch.utils.data.DataLoader(dataset=dataset[x],
                                                        batch_size=opt.BATCH_SIZE,
                                                        shuffle=shuffle[x],
                                                        num_workers=int(opt.workers),
                                                        drop_last=drop_last_batch[x])
    return dataset, dataloader

    


class AddMask():
    def __init__(self, config):
        self.flist = list(glob.glob(config.TRAIN_MASK_FLIST + '/*.jpg')) + list(glob.glob(config.TRAIN_MASK_FLIST + '/*.png'))
        self.flist.sort()
        self.mask_set = []
        self.mask_type = config.MASK_TYPE
        for scale in config.SCALES:
            for mask_index in range(scale*4, (scale+1)*4):
                mask = Image.open(self.flist[mask_index])
                mask = transforms.Resize(config.INPUT_SIZE, interpolation=Image.NEAREST)(mask)
                # mask = (mask > 0).astype(np.uint8) * 255
                self.mask_set.append(transforms.ToTensor()(mask))

    def append_mask(self, batch):
        masks = []
        imgs = []
        label = []
        clsnames = []
        xs = []
        ys = []
        for i in range(len(batch)):
            img, target, clsname, x, y = batch[i]
            imgs.append(img)
            masks.append(random.choice(self.mask_set))
            label.append(target)
            clsnames.append(clsname)
            xs.append(x)
            ys.append(y)
        imgs = torch.stack(imgs, dim=0)
        mask_batch = torch.stack(masks, dim=0)
        label = torch.FloatTensor(label)
        if self.mask_type == 0:
            mask_batch = None
        return imgs, mask_batch, label, clsnames, xs, ys
    
def get_custom_anomaly_dataset(subset, nrm_cls):
    nrm_cls_idx = subset.class_to_idx[nrm_cls]
    idx_to_class = {v: k for k, v in subset.class_to_idx.items()}
    new_targets = [0 if x == nrm_cls_idx else 1 for x in subset.targets]
    new_samples = [(x[0], 0 if x[1] == nrm_cls_idx else 1) for x in subset.samples]
    subset.class_name = [idx_to_class[x] for x in subset.targets]
    subset.targets = new_targets
    subset.samples = new_samples
    subset.imgs = new_samples
    return subset
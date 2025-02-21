import torch
import torch.nn as nn
from torch.nn import functional as F
import geomloss
from torch.utils.data import DataLoader
from torchvision import transforms
from model.resnet import wide_resnet50_2, resnet50, wide_resnet101_2, resnet18
from model.de_resnet import de_wide_resnet50_2, de_resnet50, de_wide_resnet101_2, de_resnet18
from data.DL_RD import CustomDataset
from data.DL_Contrast import train_dataset, test_dataset, AugmentationParams, SimplexNoiseParams

import os

def get_loaders(params):
    """
        Returns the train and test loader given the param config dict 
    """
    transform_fn = transforms.Compose([
                transforms.RandomHorizontalFlip(p=params.get("flip", 0.5)),
                transforms.RandomVerticalFlip(p=params.get("flip", 0.5)),
    ])
    train_data = CustomDataset(
        meta_file=params["meta_path"] + "train_metadata.json", 
        data_path=params["data_path"], 
        transform_fn=transform_fn, 
        resize_dim=(params["resize_x"], params["resize_y"]), 
        channels=params.get("channels", 3)
    )
    train_loader = DataLoader(
        train_data, 
        batch_size=params["batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_data = CustomDataset(
        meta_file=params["meta_path"] + "test_metadata.json",
        data_path=params["data_path"], 
        transform_fn=None, 
        resize_dim=(params["resize_x"], params["resize_y"]),
        channels=params.get("channels", 3)
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

def get_loaders_proj(params, test=False):
    """
        Returns the train and test loader for the Revisiting RD model given the param config dict 
    """
    test_data = test_dataset(
        meta_file=params["meta_path_test"],
        data_path=params["data_path"], 
        resize_dim=(params["resize_x"], params["resize_y"]),
        in_channels=params.get("channels", 3)
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=test)
    # only return train loader if not in "test" mode (testing or inference)
    if not test:
        aug_params = AugmentationParams(
            **{k: v for k, v in params.items() 
            if k in ['p_flip', 'p_rotate', 'p_crop', 'p_noise', 'p_blur']}
        )

        simplex_params = SimplexNoiseParams(
            **{k: v for k, v in params.items() 
            if k in ['p_simplex', 'simplex_scale', 'simplex_noise']}
        )

        train_data = train_dataset(
            meta_file=params["meta_path_train"], 
            data_path=params["data_path"], 
            transform_fn=params.get("transforms", False),
            aug_params=aug_params,
            simplex_params=simplex_params,
            resize_dim=(params["resize_x"], params["resize_y"]), 
            in_channels=params.get("channels", 3)
        )

        train_loader = DataLoader(
            train_data, 
            batch_size=params["batch_size"], 
            shuffle=True,
            num_workers=params.get("num_workers", 1),
            pin_memory=params.get("num_workers", 1) != 1,
            persistent_workers=params.get("num_workers", 1) != 1,
            # prefetch factor 1 if workers 1, else 4
            prefetch_factor=1 if params.get("num_workers", 1) == 1 else 4
        )
        
        return train_loader, test_loader
    else:
        return test_loader

def create_model(architecture: str = "wide_resnet50_2", bn_attention: bool = True, in_channels: int = 3):
    """
    Return model corresponding to the specified architecture and whether to use attention or not in the bottleneck
    """
    if architecture == "wide_resnet50_2":
        encoder, bn = wide_resnet50_2(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_wide_resnet50_2(pretrained=False)
    elif architecture == "resnet50":
        encoder, bn = resnet50(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_resnet50(pretrained=False)
    elif architecture == "resnet18":
        encoder, bn = resnet18(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_resnet18(pretrained=False)
    elif architecture == "wide_resnet101_2":
        encoder, bn = wide_resnet101_2(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_wide_resnet101_2(pretrained=False)
    elif architecture == "asym":
        encoder, bn = wide_resnet101_2(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_wide_resnet50_2(pretrained=False)
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
    return encoder, bn, decoder

def create_model_proj(architecture: str = "wide_resnet50_2", bn_attention: bool = True, in_channels: int = 3):
    """
    Return model corresponding to the specified architecture and whether to use attention or not in the bottleneck
    """
    proj_layer =  MultiProjectionLayer(base=64)
    if architecture == "wide_resnet50_2":
        encoder, bn = wide_resnet50_2(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_wide_resnet50_2(pretrained=False)
    elif architecture == "resnet50":
        encoder, bn = resnet50(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_resnet50(pretrained=False)
    elif architecture == "resnet18":
        encoder, bn = resnet18(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_resnet18(pretrained=False)
    elif architecture == "wide_resnet101_2":
        encoder, bn = wide_resnet101_2(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_wide_resnet101_2(pretrained=False)
    elif architecture == "asym":
        encoder, bn = wide_resnet101_2(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_wide_resnet50_2(pretrained=False)
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
    return encoder, bn, decoder, proj_layer

def get_optimizer(config, model_params = None):
    if str(config.get("optimizer", None)).upper() == "ADAM":
        return torch.optim.Adam(
            model_params, 
            lr=config["learning_rate"], 
            betas=(config.get("beta1", 0.5), config.get("beta2", 0.999)),
        )
    elif str(config.get("optimizer", None)).upper() == "SGD":
        return torch.optim.SGD(
            model_params,
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            dampening=config.get("dampening", 0),
        )
    elif str(config.get("optimizer", None)).upper() == "ADAMW":
        return torch.optim.AdamW(
            model_params,
            lr=config["learning_rate"],
            betas=(config.get("beta1", 0.5), config.get("beta2", 0.999)),
            weight_decay=config["weight_decay"]
        )
    else:
        print("[ERROR] UNKOWN OPTIMIZER / NO OPTIMIZER CHOSEN")
        return None

class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''
    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, in_c//4, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//4),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//4, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, out_c, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )
    def forward(self, x):
        return self.proj(x)
    
class MultiProjectionLayer(nn.Module):
    def __init__(self, base = 64):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)
    def forward(self, features, features_noise = False):
        if features_noise is not False:
            return ([self.proj_a(features_noise[0]),self.proj_b(features_noise[1]),self.proj_c(features_noise[2])], \
                  [self.proj_a(features[0]),self.proj_b(features[1]),self.proj_c(features[2])])
        else:
            return [self.proj_a(features[0]),self.proj_b(features[1]),self.proj_c(features[2])]

def loss_function(a, b, weights = [1.0,1.0,1.0]):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += weights[item] * torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss


class CosineReconstruct(nn.Module):
    def __init__(self):
        super(CosineReconstruct, self).__init__()
    def forward(self, x, y):
        return torch.mean(1 - torch.nn.CosineSimilarity()(x, y))

'''
class Revisit_RDLoss(nn.Module):
    """
    receive multiple inputs feature
    return multi-task loss:  SSOT loss, Reconstruct Loss, Contrast Loss
    """
    def __init__(self, reconstruct_weight, contrast_weight, ssot_weight, consistent_shuffle = True):
        super(Revisit_RDLoss, self).__init__()
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05, \
                              reach=None, diameter=10000000, scaling=0.9, \
                                truncate=5, cost=None, kernel=None, cluster_scale=None, \
                                  debias=True, potentials=False, verbose=False, backend='tensorized')
        self.reconstruct = CosineReconstruct()       
        self.contrast = torch.nn.CosineEmbeddingLoss(margin = 0.5)
        self.ssot_weight = ssot_weight
        self.reconstruct_weight = reconstruct_weight
        self.contrast_weight = contrast_weight

    def forward(self, noised_feature, projected_noised_feature, projected_normal_feature):
        """
        noised_feature : output of encoder at each_blocks : [noised_feature_block1, noised_feature_block2, noised_feature_block3]
        projected_noised_feature: list of the projection layer's output on noised_features, projected_noised_feature = projection(noised_feature)
        projected_normal_feature: list of the projection layer's output on normal_features, projected_normal_feature = projection(normal_feature)
        """
        current_batchsize = projected_normal_feature[0].shape[0]

        target = -torch.ones(current_batchsize).to('cuda')

        normal_proj1 = projected_normal_feature[0]
        normal_proj2 = projected_normal_feature[1]
        normal_proj3 = projected_normal_feature[2]
        # shuffling samples order for caculating pair-wise loss_ssot in batch-mode , (for efficient computation)
        shuffle_index = torch.randperm(current_batchsize, device=normal_proj1.device)
        # Shuffle the feature order of samples in each block
        shuffle_1 = normal_proj1[shuffle_index]
        shuffle_2 = normal_proj2[shuffle_index]
        shuffle_3 = normal_proj3[shuffle_index]

        abnormal_proj1, abnormal_proj2, abnormal_proj3 = projected_noised_feature
        noised_feature1, noised_feature2, noised_feature3 = noised_feature
        loss_ssot = self.sinkhorn(torch.softmax(normal_proj1.view(normal_proj1.shape[0], -1), -1), torch.softmax(shuffle_1.view(shuffle_1.shape[0], -1),-1)) +\
               self.sinkhorn(torch.softmax(normal_proj2.view(normal_proj2.shape[0], -1),-1),  torch.softmax(shuffle_2.view(shuffle_2.shape[0], -1),-1)) +\
               self.sinkhorn(torch.softmax(normal_proj3.view(normal_proj3.shape[0], -1),-1),  torch.softmax(shuffle_3.view(shuffle_3.shape[0], -1),-1))
        loss_reconstruct = self.reconstruct(abnormal_proj1, normal_proj1)+ \
                   self.reconstruct(abnormal_proj2, normal_proj2)+ \
                   self.reconstruct(abnormal_proj3, normal_proj3)
        loss_contrast = self.contrast(noised_feature1.view(noised_feature1.shape[0], -1), normal_proj1.view(normal_proj1.shape[0], -1), target = target) +\
                           self.contrast(noised_feature2.view(noised_feature2.shape[0], -1), normal_proj2.view(normal_proj2.shape[0], -1), target = target) +\
                           self.contrast(noised_feature3.view(noised_feature3.shape[0], -1), normal_proj3.view(normal_proj3.shape[0], -1), target = target)
        return (self.ssot_weight * loss_ssot + self.reconstruct_weight * loss_reconstruct + self.contrast_weight * loss_contrast)/1.11
'''
class Revisit_RDLoss(nn.Module):
    def __init__(self, reconstruct_weight, contrast_weight, ssot_weight, consistent_shuffle=True):
        super(Revisit_RDLoss, self).__init__()

        self.sinkhorn = geomloss.SamplesLoss(
            loss='sinkhorn',
            p=2,
            blur=0.05,
            diameter=None,      # set to None for increased performance
            scaling=0.9,        # decreased from original (0.95) to 0.9 for performance
            backend='tensorized',   # set to 'tensorized' as opposed to 'auto' for 'better gpu utilization'
            truncate=5, 
            debias=True
        )
        self.reconstruct = CosineReconstruct()
        self.contrast = torch.nn.CosineEmbeddingLoss(margin=0.5)
        self.ssot_weight = ssot_weight
        self.reconstruct_weight = reconstruct_weight
        self.contrast_weight = contrast_weight

    def forward(self, noised_feature, projected_noised_feature, projected_normal_feature):
        current_batchsize = projected_normal_feature[0].shape[0]
        
        target = -torch.ones(current_batchsize, device=projected_normal_feature[0].device)
        
        # pre compute softmax once for each feature
        normal_proj1 = torch.softmax(projected_normal_feature[0].view(current_batchsize, -1), -1)
        normal_proj2 = torch.softmax(projected_normal_feature[1].view(current_batchsize, -1), -1)
        normal_proj3 = torch.softmax(projected_normal_feature[2].view(current_batchsize, -1), -1)

        shuffle_index = torch.randperm(current_batchsize, device=normal_proj1.device)
        
        shuffle_1 = normal_proj1[shuffle_index]
        shuffle_2 = normal_proj2[shuffle_index]
        shuffle_3 = normal_proj3[shuffle_index]

        abnormal_proj1, abnormal_proj2, abnormal_proj3 = projected_noised_feature
        noised_feature1, noised_feature2, noised_feature3 = noised_feature

        loss_ssot = (
            self.sinkhorn(normal_proj1, shuffle_1) +
            self.sinkhorn(normal_proj2, shuffle_2) +
            self.sinkhorn(normal_proj3, shuffle_3)
        )

        loss_reconstruct = (
            self.reconstruct(abnormal_proj1, projected_normal_feature[0]) +
            self.reconstruct(abnormal_proj2, projected_normal_feature[1]) +
            self.reconstruct(abnormal_proj3, projected_normal_feature[2])
        )

        # pre flatten tensors for contrast loss
        nf1_flat = noised_feature1.view(noised_feature1.shape[0], -1)
        nf2_flat = noised_feature2.view(noised_feature2.shape[0], -1)
        nf3_flat = noised_feature3.view(noised_feature3.shape[0], -1)
        
        pnf1_flat = projected_normal_feature[0].view(projected_normal_feature[0].shape[0], -1)
        pnf2_flat = projected_normal_feature[1].view(projected_normal_feature[1].shape[0], -1)
        pnf3_flat = projected_normal_feature[2].view(projected_normal_feature[2].shape[0], -1)

        loss_contrast = (
            self.contrast(nf1_flat, pnf1_flat, target) +
            self.contrast(nf2_flat, pnf2_flat, target) +
            self.contrast(nf3_flat, pnf3_flat, target)
        )

        return (self.ssot_weight * loss_ssot + 
                self.reconstruct_weight * loss_reconstruct + 
                self.contrast_weight * loss_contrast) / 1.11

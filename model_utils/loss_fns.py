import torch
import numpy as np
import torch.nn.functional as F

def loss_function(a, b, weights):
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

def loss_function_margin(a, b, weights, margin=0.1):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        similarity = cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1))
        loss += weights[item] * torch.mean(torch.clamp(1 - similarity - margin, min=0))
    return loss

def loss_function_l2(a, b, weights, l2_lambda=0.01):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        similarity = cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1))
        loss += weights[item] * torch.mean(1 - similarity)
        loss += l2_lambda * torch.mean(torch.norm(b[item], p=2, dim=1))
    return loss

def loss_function_noise(a, b, weights, noise_std=0.1):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_noisy = a[item] + torch.randn_like(a[item]) * noise_std
        similarity = cos_loss(a_noisy.view(a_noisy.shape[0],-1), b[item].view(b[item].shape[0],-1))
        loss += weights[item] * torch.mean(1 - similarity)
    return loss

def loss_function_mse(a, b, weights, mse_weight=0.2):
    cos_loss = torch.nn.CosineSimilarity()
    mse_loss = torch.nn.MSELoss()
    loss = 0
    for item in range(len(a)):
        a_flat = a[item].view(a[item].shape[0],-1)
        b_flat = b[item].view(b[item].shape[0],-1)
        cos_sim = cos_loss(a_flat, b_flat)
        mse = mse_loss(a_flat, b_flat)
        loss += weights[item] * (torch.mean(1 - cos_sim) + mse_weight * mse)
    return loss

def focal_cos_loss(cos_sim, gamma=2.0, alpha=0.25):
    return alpha * (1 - cos_sim)**gamma

def loss_function_focal(a, b, weights, gamma=2.0, alpha=0.25):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        similarity = cos_loss(a[item].view(a[item].shape[0],-1), b[item].view(b[item].shape[0],-1))
        loss += weights[item] * torch.mean(focal_cos_loss(similarity, gamma, alpha))
    return loss
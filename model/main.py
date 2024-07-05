import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from test import evaluation, visualization, test
from torch.nn import functional as F

import warnings

# ignore deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# NOTE: a and b are lists of feature maps
# inputs (a) = encoder(img)
# outputs (b) = decoder(bn(inputs))
# the models are designed to output features (output of each convolution) as a list on a forward pass
def loss_function(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    # iterate over each feature map set (outputs from a conv layer)
    for item in range(len(a)):
        # a[item] has form BxCxHxW reshaped to Bx(C*H*W)
        # calculate cosine similarity loss between the feature maps of a layer
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    
    return loss

# i haven't seen this used here...
def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss

def train(_class_):
    """
        Train the model on the specified class (for MVTEC, carpet, bottle, etc.). TODO: REMOVE THIS AND PASS HYPERPARAMETERS YAML
    """
    print(_class_)
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # data loaders
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = './mvtec/' + _class_ + '/train'
    test_path = './mvtec/' + _class_
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # model
    encoder, bn = wide_resnet50_2(pretrained=True)  # TODO: pass model arch as argument to train()
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()  # NOTE: freeze encoder
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    # optimizer is adam here. TODO: betas?
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            
            inputs = encoder(img)           # encoder(img) is the output of the encoder
            outputs = decoder(bn(inputs))   # decoder(bn(inputs)) is the output of the decoder
            
            loss = loss_function(inputs, outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))   # for the epoch print the mean loss
        
        # every 10 epochs, evaluate the model and save its state
        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)    # NOTE: sepearte evaluation method
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}'.format(auroc_px, auroc_sp, aupro_px))
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
    return auroc_px, auroc_sp, aupro_px

if __name__ == '__main__':
    setup_seed(111)
    # classes
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']

    total_start_time = time.time()  # start timer for total
    training_times = {}

    for item in item_list:
        start_time = time.time()  # start per class timer
        train(item)
        end_time = time.time()
        training_duration = end_time - start_time
        training_times[item] = training_duration
        print(f"Training time for {item}: {training_duration:.2f} seconds")

    total_end_time = time.time()  # end total timer
    total_training_duration = total_end_time - total_start_time

    print("Training times for all items:", training_times)
    print(f"Total training time: {total_training_duration:.2f} seconds")
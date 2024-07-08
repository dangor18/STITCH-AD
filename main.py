import time
import warnings
import random
import os
import yaml

import torch
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np

from model_utils.resnet import resnet18, resnet34, resnet50, wide_resnet50_2, resnet101, resnet152, resnext101_32x8d, resnext50_32x4d, wide_resnet101_2
from model_utils.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50, resnet101, resnet152, resnext101_32x8d, resnext50_32x4d, de_wide_resnet101_2
from model_utils.test import evaluation, visualization, test

from data.dataset_loader import CustomDataset, custom_standardize_transform, custom_pscaling_transform

# ignore deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)

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


resnet_models = {
    'resnet18': (resnet18, de_resnet18),
    'resnet34': (resnet34, de_resnet34),
    'resnet50': (resnet50, de_resnet50),
    'resnet101': (resnet101, resnet101),
    'resnext50_32x4d': (resnext50_32x4d, resnext50_32x4d),
    'resnext101_32x8d': (resnext101_32x8d, resnext101_32x8d),
    'wide_resnet50_2': (wide_resnet50_2, de_wide_resnet50_2),
    'wide_resnet101_2': (wide_resnet101_2, de_wide_resnet101_2)
}

def train(input_channels, num_epochs, train_loader, test_loader, learning_rate, weight_decay, model_path, device, arch):
    #convh = ConvH(input_channel=input_channels)
    #convh.to(device)
    #encoder_func, decoder_func = resnet_models[arch]
    
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    #optimizer = torch.optim.Adam(list(decoder.parameters())+list(convh.parameters())+list(bn.parameters()), lr=learning_rate, weight_decay=weight_decay, betas=(0.5,0.999))
    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate,weight_decay=0, betas=(0.5,0.999))

    for epoch in range(num_epochs):
        #convh.train()
        bn.train()
        decoder.train()
        loss_list = []
        for noise_imgs, images, _ in train_loader:
            #noise_imgs = noise_imgs.to(device)
            images = images.to(device)
            
            #inputs = encoder(convh(images))
            inputs = encoder(images)
            outputs = decoder(bn(inputs))
            
            loss = loss_function(inputs, outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
        print(f'epoch [{epoch + 1}/{num_epochs}], loss:{np.mean(loss_list):.4f}')
        
        if (epoch + 1) % 10 == 0:
            auroc_overall, auroc_case1, auroc_case2 = evaluation(encoder, bn, decoder, test_loader, device, plot_results=True, n_plot_per_class=50)
            print(f"Epoch {epoch + 1} Evaluation:")
            if auroc_overall is not None:
                print(f'Overall AUROC: {auroc_overall:.3f}')
            if auroc_case1 is not None:
                print(f'Case 1 AUROC: {auroc_case1:.3f}')
            if auroc_case2 is not None:
                print(f'Case 2 AUROC: {auroc_case2:.3f}')

            with open("log.txt", "a") as file:
                file.write(f'\nepoch [{epoch + 1}/{num_epochs}], loss:{np.mean(loss_list):.4f}')
                file.write(f"\nCase 1 AUROC: {auroc_case1:.3f}, Case 2 AUROC: {auroc_case2:.3f}, Overall AUROC: {auroc_overall:.3f}")
            
            torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, model_path)

    return auroc_overall, auroc_case1, auroc_case2

if __name__ == '__main__':
    setup_seed(111)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # hyper-parameters
    try:
        with open("model_config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        num_epochs = cfg["num_epochs"]
        batch_size = cfg["batch_size"]
        learning_rate = cfg["learning_rate"]
        weight_decay = cfg["weight_decay"]
        data_path = cfg["data_path"]                # path to data
        data_stats_path = cfg["data_stats_path"]    # where to load the stats from to preprocess
        model_path = cfg["model_path"]              # path to save states
        noise_factor = cfg["noise_factor"]
        p = cfg["p"]
        resize_x = cfg["resize_x"]
        resize_y = cfg["resize_y"]
        channels = cfg["channels"]
        print(f'Config loaded: Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}, Weight decay: {weight_decay}, Noise factor: {noise_factor}, p: {p}, Resize : ({resize_x}, {resize_y})')
    except Exception as e:
        print("Error reading config file: \n", e)
        exit()

    # load data_stats from data_stats_path
    mean_path = os.path.join(data_stats_path, "train_means.npy")
    sdv_path = os.path.join(data_stats_path, "train_sdv.npy")
    p_path = os.path.join(data_stats_path, "train_percentiles.npy")
    # transform
    transform_stand = custom_standardize_transform(np.load(mean_path), np.load(sdv_path), device)
    #transform_perc = custom_pscaling_transform(np.load(p_path), device)
    # data loaders
    test_data = CustomDataset(data_path + "val", transform_stand, (resize_x, resize_y))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    train_data = CustomDataset(data_path + "train", transform_stand, (resize_x, resize_y), noise_factor=noise_factor, p=p)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    archs = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']
    #for a in archs:
    # train model
    start_time = time.time()  # start timer for total training time
    train(channels, num_epochs, train_loader, test_loader, learning_rate, weight_decay, model_path, device, 'wide_resnet50_2')
    end_time = time.time()
    training_duration = end_time - start_time
    
    print(f"Training time for {archs[-2]}: {training_duration/(60*60):.2f} hours for {num_epochs} epochs")
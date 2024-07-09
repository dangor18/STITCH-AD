import time
import warnings
import random
import os
import yaml

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np

from model_utils.resnet import wide_resnet50_2
from model_utils.de_resnet import de_wide_resnet50_2
from model_utils.test import evaluation

from data.dataset_loader import CustomDataset

# ignore deprecation warnings
warnings.filterwarnings('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_function(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def train(input_channels, num_epochs, train_loader, test_loader, learning_rate, weight_decay, model_path, device):
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, weight_decay=weight_decay, betas=(0.5,0.999))

    for epoch in range(num_epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for input in train_loader:
            noisy_images = input["noisy_image"].to(device)
            images = input["image"].to(device)
            
            inputs = encoder(images)
            outputs = decoder(bn(inputs))
            
            loss = loss_function(inputs, outputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
        print(f'epoch [{epoch + 1}/{num_epochs}], loss:{np.mean(loss_list):.4f}')
    
        if (epoch + 1) % 10 == 0:
            auroc_overall, auroc_case1, auroc_case2 = evaluation(encoder, bn, decoder, test_loader, device)
            print(f"Epoch {epoch + 1} Evaluation:")
            print(f'Overall AUROC: {auroc_overall:.3f}, Case 1 AUROC: {auroc_case1:.3f}, Case 2 AUROC: {auroc_case2:.3f}')

            with open("log.txt", "a") as file:
                file.write(f'\nepoch [{epoch + 1}/{num_epochs}], loss:{np.mean(loss_list):.4f}')
                file.write(f"\nCase 1 AUROC: {auroc_case1:.3f}, Case 2 AUROC: {auroc_case2:.3f}, Overall AUROC: {auroc_overall:.3f}")
            
            torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, model_path)

    return auroc_overall, auroc_case1, auroc_case2

if __name__ == '__main__':
    setup_seed(111)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load configuration
    with open("model_config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # Create data loaders
    train_data = CustomDataset(cfg["meta_path"] + "train_metadata.json", cfg["data_path"], None, (cfg["resize_x"], cfg["resize_y"]), noise_factor=cfg["noise_factor"], p=cfg["p"])
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True, pin_memory=True, num_workers=4)

    test_data = CustomDataset(cfg["meta_path"] + "test_metadata.json", cfg["data_path"], None, (cfg["resize_x"], cfg["resize_y"]))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Train model
    start_time = time.time()
    auroc_overall, auroc_case1, auroc_case2 = train(
        cfg["channels"], 
        cfg["num_epochs"], 
        train_loader, 
        test_loader, 
        cfg["learning_rate"], 
        cfg["weight_decay"], 
        cfg["model_path"], 
        device
    )
    end_time = time.time()
    training_duration = end_time - start_time
    
    print(f"Training completed in {training_duration/3600:.2f} hours")
    print(f"Final results - Overall AUROC: {auroc_overall:.3f}, Case 1 AUROC: {auroc_case1:.3f}, Case 2 AUROC: {auroc_case2:.3f}")
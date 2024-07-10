import time
import warnings
import random
import os
import yaml

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np

from model_utils.resnet import wide_resnet50_2, resnet50
from model_utils.de_resnet import de_wide_resnet50_2, de_resnet50
from model_utils.test import evaluation

from data.dataset_loader import CustomDataset

from tqdm import tqdm

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

def train_single_epoch(encoder, bn, decoder, optimizer, train_loader, device):
    bn.train()
    decoder.train()
    loss_sum = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for input in progress_bar:
        noisy_images = input["noisy_image"].to(device)
        images = input["image"].to(device)
        
        inputs = encoder(images)
        outputs = decoder(bn(inputs))
        
        loss = loss_function(inputs, outputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        num_batches += 1
    
    return loss_sum / num_batches if num_batches > 0 else 0.0

def train(input_channels, num_epochs, train_loader, test_loader, learning_rate, weight_decay, model_path, device):
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))

    for epoch in range(num_epochs):
        start_time = time.time()
        avg_loss = 0
        avg_loss = train_single_epoch(encoder, bn, decoder, optimizer, train_loader, device)
        
        end_time = time.time()
        training_duration = end_time - start_time
        print(f'EPOCH [{epoch + 1}/{num_epochs}], LOSS: {avg_loss:.4f}, TIME: {training_duration/60:.2f} MINUTES')
    
        if (epoch + 1) % 10 == 0:
            with open("log.txt", "a") as file:
                file.write(f'\nEPOCH [{epoch + 1}/{num_epochs}], LOSS: {avg_loss:.4f}, TIME: {training_duration/60:.2f} MINUTES')

            auroc_overall = evaluation(encoder, bn, decoder, test_loader, device)

            print(f"Epoch {epoch + 1} Evaluation:")
            print(f'Overall AUROC: {auroc_overall:.3f}')

            with open("log.txt", "a") as file:
                file.write(f"\n- OVERALL AUROC: {auroc_overall:.3f}")
            
            torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, model_path)
    
    #auroc_overall, auroc_case1, auroc_case2 = evaluation(encoder, bn, decoder, test_loader, device, plot_results=True, n_plot_per_class=50)

    return auroc_overall, auroc_case1, auroc_case2

if __name__ == '__main__':
    setup_seed(111)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Load configuration
    with open("model_config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # Create data loaders
    print("[INFO] LOADING DATA...")
    train_data = CustomDataset(cfg["meta_path"] + "train_metadata.json", cfg["data_path"], None, (cfg["resize_x"], cfg["resize_y"]), noise_factor=cfg["noise_factor"], p=cfg["p"])
    train_loader = DataLoader(
        train_data, 
        batch_size=cfg["batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_data = CustomDataset(cfg["meta_path"] + "test_metadata.json", cfg["data_path"], None, (cfg["resize_x"], cfg["resize_y"]))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print("[INFO] STARTING TRAINING...")
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
    num_epochs = cfg["num_epochs"]
    print(f"\n[INFO] TRAINING COMPLETED {num_epochs} EPOCHS IN {training_duration/3600:.2f} hours")
    print(f"[INFO] FINAL RESULTS:\n OVERALL AUROC: {auroc_overall:.3f}, Case 1 AUROC: {auroc_case1:.3f}, Case 2 AUROC: {auroc_case2:.3f}")
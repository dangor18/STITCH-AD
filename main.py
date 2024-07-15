import time
import warnings
import random
import os
import yaml
import argparse
import optuna

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import numpy as np

from model_utils.resnet import wide_resnet50_2, resnet50, wide_resnet101_2, ChannelReductionCBAM
from model_utils.de_resnet import de_wide_resnet50_2, de_resnet50, de_wide_resnet101_2
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
    torch.backends.cudnn.benchmark = True

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

def create_model(architecture: str = "wide_resnet50_2", attention: bool = True):
    """
    Return model corresponding to the specified architecture and whether to use attention or not in the bottleneck
    """
    if architecture == "wide_resnet50_2":
        encoder, bn = wide_resnet50_2(pretrained=True, attention=attention)
        decoder = de_wide_resnet50_2(pretrained=False)
    elif architecture == "resnet50":
        encoder, bn = resnet50(pretrained=True, attention=attention)
        decoder = de_resnet50(pretrained=False)
    elif architecture == "wide_resnet101_2":
        encoder, bn = wide_resnet101_2(pretrained=True, attention=attention)
        decoder = de_wide_resnet101_2(pretrained=False)
    elif architecture == "asym":
        encoder, bn = wide_resnet101_2(pretrained=True, attention=attention)
        decoder = de_wide_resnet50_2(pretrained=False)
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
    return encoder, bn, decoder

def train_tuning(params, trial):
    """
    Train with hyperparameter tuning (no logs, no model saves, no printing to console)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders(params)
    
    #pre_conv = ChannelReductionCBAM(in_channels=params["channels"], ratio=2, emphasis_channel=0, weight=params["dem_weight"])
    encoder, bn, decoder = create_model(params["architecture"], attention=params["attention"])
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(
        list(decoder.parameters()) + list(bn.parameters()), 
        lr=params["learning_rate"], 
        betas=(params.get("beta1", 0.5), params.get("beta2", 0.999)),
        weight_decay=params["weight_decay"]
    )

    best_auroc = 0
    # train loop
    for epoch in range(params["num_epochs"]):
        bn.train()
        decoder.train()
        loss_sum = 0.0
        num_batches = 0
        
        for input in train_loader:
            images = input["image"].to(device)
            
            inputs = encoder(images)
            outputs = decoder(bn(inputs))
            loss = loss_function(inputs, outputs, weights=[params.get("low_weight", 1)] + [1 for _ in range(len(inputs) - 1)])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            total_auroc = evaluation(encoder, bn, decoder, test_loader, device)
            
            if total_auroc > best_auroc:
                best_auroc = total_auroc
            
            if trial:  # prune training if necessary (bad params)
                trial.report(total_auroc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_auroc

def train_normal(params, train_loader, test_loader, device):
    """
    Train the model (without hyperparameter) tuning on orchard patch data with the specified parameters and return the best overall AUROC score
    """
    #pre_conv = ChannelReductionCBAM(in_channels=params["channels"], ratio=2, emphasis_channel=0, weight=params["dem_weight"])
    encoder, bn, decoder = create_model(params["architecture"], attention=params["attention"])
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = decoder.to(device)

    # nb add params here for pre_conv
    optimizer = torch.optim.Adam(
        list(decoder.parameters()) + list(bn.parameters()), 
        lr=params["learning_rate"], 
        betas=(params.get("beta1", 0.5), params.get("beta2", 0.999)),
        weight_decay=params["weight_decay"]
    )

    scaler = GradScaler()

    best_auroc = 0
    print("[INFO] TRAINING MODEL...")
    # train loop
    for epoch in range(params["num_epochs"]):
        bn.train()
        decoder.train()
        loss_sum = 0.0
        num_batches = 0
        
        train_data = tqdm(train_loader)

        for input in train_data:
            images = input["image"].to(device)
            with autocast():
                inputs = encoder(images)
                outputs = decoder(bn(inputs))
                loss = loss_function(inputs, outputs, weights=[params.get("low_weight", 1)] + [1 for _ in range(len(inputs) - 1)])
            
            optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_sum += loss.item()
            num_batches += 1

        avg_loss = loss_sum / num_batches
        
        print(f"EPOCH {epoch + 1}, LOSS: {avg_loss:.3f}")
        with open(params["log_path"], "a") as log_file:
            log_file.write(f"\nEPOCH {epoch + 1}, LOSS: {avg_loss:.3f}\n")
        # evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            total_auroc = evaluation(encoder, bn, decoder, test_loader, device, params["log_path"], params.get("low_weight", 1))
            print(f"EPOCH {epoch + 1}, LOSS: {avg_loss:.3f}, OVERALL AUROC: {total_auroc:.3f}")
            
            if total_auroc > best_auroc:
                best_auroc = total_auroc
                # save model
                print(f"[INFO] NEW BEST. SAVING MODEL TO {params['model_path']}...")
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, params["model_path"])

    return best_auroc

def get_loaders(params):
    transform_fn = transforms.Compose([
                transforms.RandomHorizontalFlip(p=params["p_flip"]),
                transforms.RandomVerticalFlip(p=params["p_flip"]),
                #transforms.ColorJitter(brightness=0.0, contrast=params["contrast"], saturation=0.0, hue=0.0),
                #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    ])  
    train_data = CustomDataset(
            params["meta_path"] + "train_metadata.json", 
            params["data_path"], 
            transform_fn, 
            (params["resize_x"], params["resize_y"]), 
            noise_factor=params["noise_factor"], 
            p=params["p_flip"],
            norm_choice=params["norm_choice"]
        )
    train_loader = DataLoader(
            train_data, 
            batch_size=params["batch_size"], 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
    )
    test_data = CustomDataset(
            params["meta_path"] + "test_metadata.json",
            params["data_path"], 
            None, 
            (params["resize_x"], params["resize_y"]),
            norm_choice=params["norm_choice"]
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

# objective function for optuna
def objective(trial):
    # open config
    with open("model_config.yaml", "r") as ymlfile:
        params = yaml.safe_load(ymlfile)

    # new param suggestions
    params["learning_rate"] = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    params["batch_size"] = trial.suggest_categorical("batch_size", [8, 12, 16, 20, 32])
    params["weight_decay"] = trial.suggest_uniform("weight_decay", 0, 1e-3)
    params["architecture"] = trial.suggest_categorical("architecture", ["wide_resnet50_2", "resnet50", "wide_resnet101_2", "asym"]) # asym for asymetric encoder decoder arch
    params["attention"] = trial.suggest_categorical("attention", [True, False])
    params["beta1"] = trial.suggest_uniform("beta1", 0.5, 0.9)
    params["beta2"] = trial.suggest_uniform("beta2", 0.9, 0.999)
    #params["p"] = trial.suggest_uniform("p", 0, 0.5)
    params["norm_choice"] = trial.suggest_categorical("norm_choice", ["PER_ORCHARD", "IMAGE_NET"])
    params["low_weight"] = trial.suggest_uniform("low_weight", 1, 2)
    params["dem_weight"] = trial.suggest_uniform("dem_weight", 1, 2)

    return train_tuning(params, trial)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # tune with optuna or train with default parameters from config file
    if args.tune is True:
        print("[INFO] TUNING HYPERPARAMETERS...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        #print("[INFO] BEST HYPERPARAMETERS:")
        trial = study.best_trial
        for key, val in trial.params.items():
            #print(f"{key}: {val}")
            with open("logs/hypertuning.txt") as file:
                file.write(f"{key}: {val}")
        with open("logs/hypertuning.txt") as file:
                file.write(f"[INFO] BEST AUROC: {trial.value:.3f}")
        #print(f"[INFO] BEST AUROC: {trial.value:.3f}")
    else:   # else parameters from config file
        with open("model_config.yaml", "r") as config_file:
            params = yaml.safe_load(config_file)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("[INFO] DEVICE:", device) 
        # create data loaders
        print("[INFO] LOADING DATA...")
        train_loader, test_loader = get_loaders(params)
        
        # train
        best_auroc = train_normal(params, train_loader, test_loader, device)
        print(f"[INFO] BEST OVERALL AUROC: {best_auroc:.3f}")
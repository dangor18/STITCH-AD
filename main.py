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
#warnings.filterwarnings('ignore')

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
    elif architecture == "wide_resnet101_2":
        encoder, bn = wide_resnet101_2(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_wide_resnet101_2(pretrained=False)
    elif architecture == "asym":
        encoder, bn = wide_resnet101_2(pretrained=True, attention=bn_attention, in_channels=in_channels)
        decoder = de_wide_resnet50_2(pretrained=False)
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
    return encoder, bn, decoder

def get_optimizer(config, model_params = None):
    if str(config.get("optimizer", None)).upper() == "ADAM":
        return torch.optim.Adam(
            model_params, 
            lr=config["learning_rate"], 
            betas=(config.get("beta1", 0.5), config.get("beta2", 0.999)),
            weight_decay=config["weight_decay"]
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

def train_tuning(params, trial):
    """
    Train with hyperparameter tuning (no logs, no model saves, no printing to console)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders(params)
    
    #pre_conv = ChannelReductionCBAM(in_channels=params["channels"], ratio=2, emphasis_channel=0, weight=params["dem_weight"])
    encoder, bn, decoder = create_model(architecture=params["architecture"], bn_attention=params["bn_attention"], in_channels=params.get("channels", 3))
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = decoder.to(device)

    optimizer = get_optimizer(params, list(decoder.parameters()) + list(bn.parameters()))
    
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=params["lr_factor"],
        patience=params["patience"],
        verbose=True
    )

    scaler = GradScaler()

    best_auroc = 0
    # train loop
    for epoch in range(params["num_epochs"]):
        bn.train()
        decoder.train()
        
        for input in train_loader:
            images = input["image"].to(device)
            with autocast():
                inputs = encoder(images)
                outputs = decoder(bn(inputs))
                loss = loss_function(inputs, outputs, weights=params.get("weights", [1.0, 1.0, 1.0]))
            
            optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # evaluate every 3 epochs
        if (epoch + 1) % 3 == 0:
            total_auroc = evaluation(encoder, bn, decoder, test_loader, device, weights=params.get("weights", [1.0, 1.0, 1.0]))
            
            if total_auroc > best_auroc:
                best_auroc = total_auroc
            
            scheduler.step(total_auroc)

            # prune training if necessary (bad params)
            trial.report(total_auroc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_auroc

def train_normal(params, train_loader, test_loader, device):
    """
    Train the model (without hyperparameter) tuning on orchard patch data with the specified parameters and return the best overall AUROC score
    """
    #pre_conv = ChannelReductionCBAM(in_channels=params["channels"], ratio=2, emphasis_channel=0, weight=params["dem_weight"])
    encoder, bn, decoder = create_model(architecture=params["architecture"], bn_attention=params["bn_attention"], in_channels=params.get("channels", 3))
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = decoder.to(device)
 
    optimizer = get_optimizer(params, list(decoder.parameters()) + list(bn.parameters()))
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=params["lr_factor"],
        patience=params["patience"],
        verbose=True
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
                loss = loss_function(inputs, outputs, weights=params.get("weights", [1.0, 1.0, 1.0]))
            
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
            total_auroc = evaluation(encoder, bn, decoder, test_loader, device, params["log_path"], weights=params.get("weights", [1.0, 1.0, 1.0]))
            print(f"EPOCH {epoch + 1}, LOSS: {avg_loss:.3f}, OVERALL AUROC: {total_auroc:.5f}")
            
            if total_auroc > best_auroc:
                best_auroc = total_auroc
                # save model
                print(f"[INFO] NEW BEST. SAVING MODEL TO {params['model_path']}...")
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, params["model_path"])
            
            scheduler.step(total_auroc)

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
    with open("configs/model_config.yaml", "r") as ymlfile:
        params = yaml.safe_load(ymlfile)

    # new param suggestions
    params["learning_rate"] = trial.suggest_float("learning_rate", low=1e-5, high=1e-2, log=True)
    params["lr_factor"] = trial.suggest_float("lr_factor", low=0, high=0.9)
    params["patience"] = trial.suggest_int("patience", low=2, high=10, step=1)
    params["batch_size"] = trial.suggest_categorical("batch_size", [16, 24, 32, 64])
    params["weight_decay"] = trial.suggest_float("weight_decay", low=1e-6, high=1e-2, log=True)
    params["architecture"] = trial.suggest_categorical("architecture", ["wide_resnet50_2", "resnet50", "wide_resnet101_2", "asym"]) # asym for asymetric encoder decoder arch
    params["bn_attention"] = trial.suggest_categorical("bn_attention", [True, False])
    #params["optimizer"] = trial.suggest_categorical("optimizer", ["ADAM", "ADAMW", "SGD"])
    if str(params["optimizer"]).upper() == "ADAM" or str(params["optimizer"]).upper() == "ADAMW":
        params["beta1"] = trial.suggest_float("beta1", low=0.5, high=0.9999)
        params["beta2"] = trial.suggest_float("beta2", low=0.9, high=0.9999)
    elif str(params["optimizer"]).upper() == "SGD":
        params["momentum"] = trial.suggest_float("momentum", low=0.0, high=0.99)

    #params["p_flip"] = trial.suggest_float("p_flip", 0, 0.5)
    #params["norm_choice"] = trial.suggest_categorical("norm_choice", ["PER_ORCHARD", "IMAGE_NET"])
    #params["weights"] = [trial.suggest_float("weights", low=0.1, high=1.0) for _ in range(3)]
    #params["dem_weight"] = trial.suggest_uniform("dem_weight", 1, 2)

    return train_tuning(params, trial)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", default="configs/model_config.yaml", required=False)
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    args = parser.parse_args()

    config = args.config

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # tune with optuna or train with default parameters from config file
    if args.tune is True:
        print("[INFO] TUNING HYPERPARAMETERS...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=15)
        print("[INFO] BEST HYPERPARAMETERS:")
        trial = study.best_trial
        for key, val in trial.params.items():
            print(f"{key}: {val}")
        print(f"[INFO] BEST AUROC: {trial.value:.5f}")
    else:   # else parameters from config file
        with open(config, "r") as config_file:
            params = yaml.safe_load(config_file)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("[INFO] DEVICE:", device) 
        # create data loaders
        print("[INFO] LOADING DATA...")
        train_loader, test_loader = get_loaders(params)
        
        # train
        best_auroc = train_normal(params, train_loader, test_loader, device)
        print(f"[INFO] BEST OVERALL AUROC: {best_auroc:.5f}")
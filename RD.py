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
from torch.amp import autocast, GradScaler
from torchvision import transforms
import numpy as np

from model.resnet import wide_resnet50_2, resnet50, wide_resnet101_2, resnet18
from model.de_resnet import de_wide_resnet50_2, de_resnet50, de_wide_resnet101_2, de_resnet18
from model_utils.test_utils import evaluation, test
from model_utils.plots import plot_auroc
from data.DL_RD import CustomDataset, AddGaussianNoise
from model_utils.train_utils import loss_function

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

def get_loaders(params):
    transform_fn = transforms.Compose([
                #transforms.RandomHorizontalFlip(p=params["flip"]),
                #transforms.RandomVerticalFlip(p=params["flip"]),
                #transforms.RandomResizedCrop(256, scale=(params["crop_min"], 1.0)),
                #transforms.ColorJitter(contrast=(0.9, 1.1)),
                #transforms.RandomRotation(degrees=params["degrees"]),
    ])  
    train_data = CustomDataset(
        params["meta_path"] + "train_metadata.json", 
        params["data_path"], 
        transform_fn, 
        (params["resize_x"], params["resize_y"]), 
        noise_factor=0,
        p=params["flip"],
        norm_choice=params["norm_choice"],
        channels=params.get("channels", 3)
    )
    train_loader = DataLoader(
        train_data, 
        batch_size=params["batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_data = CustomDataset(
        params["meta_path"] + "test_metadata.json",
        params["data_path"], 
        None, 
        (params["resize_x"], params["resize_y"]),
        norm_choice=params["norm_choice"],
        channels=params.get("channels", 3)
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

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
    elif architecture == "efficientNet":
        encoder, bn = efficientnet_b0(pretrained=True, outblocks=[3, 5, 15], outstrides=[8, 16, 32])
        decoder = model_utils.efficientnet.decoder.de_wide_resnet50_2(pretrained=False)
    else:
        raise ValueError(f"Unknown model architecture: {architecture}")
    return encoder, bn, decoder

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

def train_tuning(params, trial):
    """
    Train with hyperparameter tuning (no logs, no model saves, no printing to console)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders(params)
    
    encoder, bn, decoder = create_model(architecture=params["architecture"], bn_attention=params["bn_attention"], in_channels=params.get("channels", 3))
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = decoder.to(device)

    optimizer = get_optimizer(params, list(decoder.parameters()) + list(bn.parameters()))
    #loss_fn = get_loss_fn(params)
    
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=params.get("step", 10),
        gamma=params["lr_factor"],
    )

    scaler = GradScaler("cuda")

    best_auroc = 0
    # train loop
    for epoch in range(params["num_epochs"]):
        bn.train()
        decoder.train()
        
        for input in train_loader:
            images = input["image"].to(device)
            with autocast(device_type="cuda"):
                inputs = encoder(images)
                outputs = decoder(bn(inputs))
                loss = loss_function(inputs, outputs, params.get("feature_weights", [1.0, 1.0, 1.0]))
            
            optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # evaluate every 10 epochs
        if (epoch + 1) % 2 == 0 and epoch + 1 > 3:
            total_auroc, _ = evaluation(encoder, bn, decoder, test_loader, device, score_weight=params.get("score_weight", 0.0), score_mode=params.get("score_mode", "a"), temp=params["feature_weights"])
            
            if total_auroc > best_auroc:
                best_auroc = total_auroc
            
            scheduler.step()

            # prune training if necessary (bad params)
            trial.report(total_auroc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_auroc

def train_normal(params, train_loader, test_loader, device):
    """
    Train the model (without hyperparameter) tuning on orchard patch data with the specified parameters and return the best overall AUROC score
    """
    encoder, bn, decoder = create_model(architecture=params["architecture"], bn_attention=params["bn_attention"], in_channels=params.get("channels", 3))
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = decoder.to(device)
    optimizer = get_optimizer(params, list(decoder.parameters()) + list(bn.parameters()))
    #loss_fn = get_loss_fn(params)
    # lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=params.get("step", 10),
        gamma=params["lr_factor"],
    )

    scaler = GradScaler("cuda")

    best_auroc = 0
    auroc_dict = {}
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
            with autocast(device_type="cuda"):
                inputs = encoder(images)
                outputs = decoder(bn(inputs))
                loss = loss_function(inputs, outputs, params.get("feature_weights", [1.0, 1.0, 1.0]))
            
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
        if (epoch + 1) % 2 == 0 and epoch + 1 > 3:
            total_auroc, orchard_auroc_dict = evaluation(encoder, bn, decoder, test_loader, device, params["log_path"], temp=params["feature_weights"],
                                                         score_weight=params.get("score_weight", 0.0))
            
            # collect aurocs for each orchard and total for plotting
            auroc_dict[epoch+1] = orchard_auroc_dict
            
            print(f"EPOCH {epoch + 1}, LOSS: {avg_loss:.3f}, OVERALL AUROC: {total_auroc:.5f}")
            
            if total_auroc > best_auroc:
                best_auroc = total_auroc
                # save model
                print(f"[INFO] NEW BEST. SAVING MODEL TO {params['model_path']}...")
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, params["model_path"])
            
            scheduler.step()
    
    test(encoder, bn, decoder, test_loader, device, params["model_path"], score_weight=params.get("score_weight", 0.0), feature_weights=params["feature_weights"], n_plot_per_class=0)
    plot_auroc(auroc_dict)
    return best_auroc

def write_to_file(study, trial):
    with open("logs/optuna_results.txt", "a") as f:
        f.write(f"Trial {trial.number}:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")
        f.write("\n")

# objective function for optuna
def objective(trial):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", default="configs/RD_config.yaml", required=False)
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    args = parser.parse_args()

    config = args.config
    # open config
    with open(config, "r") as ymlfile:
        params = yaml.safe_load(ymlfile)

    params["learning_rate"] = trial.suggest_float("learning_rate", low=1e-5, high=1e-2, log=True)
    #params["lr_factor"] = trial.suggest_float("lr_factor", low=0, high=0.5)
    params["step"] = trial.suggest_int("step", low=1, high=10)
    params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])
    #params["weight_decay"] = trial.suggest_float("weight_decay", low=1e-6, high=1e-2, log=True)
    #params["architecture"] = trial.suggest_categorical("architecture", ["wide_resnet50_2", "resnet50", "wide_resnet101_2", "asym"]) # asym for asymetric encoder decoder arch
    params["bn_attention"] = trial.suggest_categorical("bn_attention", [False, "CBAM", "SE"])
    params["beta1"] = trial.suggest_float("beta1", low=0.5, high=0.9999)
    params["beta2"] = trial.suggest_float("beta2", low=0.9, high=0.9999)

    params["feature_weight1"] = trial.suggest_float("feature_weight1", low=0.5, high=1.5)
    params["feature_weight2"] = trial.suggest_float("feature_weight2", low=0.5, high=1.5)
    params["feature_weight3"] = trial.suggest_float("feature_weight3", low=0.5, high=1.5)
    params["feature_weights"] = [params["feature_weight1"], params["feature_weight2"], params["feature_weight3"]]
    params["score_weight"] = trial.suggest_float("score_weight", low=0.0, high=0.5)

    return train_tuning(params, trial)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", default="configs/RD_config.yaml", required=False)
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    args = parser.parse_args()

    config = args.config

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    setup_seed(111)

    # tune with optuna or train with default parameters from config file
    if args.tune is True:
        print("[INFO] TUNING HYPERPARAMETERS...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=150, callbacks=[write_to_file])
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
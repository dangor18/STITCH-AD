import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json
import optuna
from argparse import ArgumentParser
import yaml
from torch.amp import autocast, GradScaler
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from model_utils.test_utils import evaluation_multi_proj
from model_utils.train_utils import MultiProjectionLayer, Revisit_RDLoss, loss_function
from data.DL_Contrast import test_dataset, train_dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(params):
    train_data = train_dataset(
        params["meta_path"] + "train_metadata.json", 
        params["data_path"], 
        (params["resize_x"], params["resize_y"]), 
    )
    train_loader = DataLoader(
        train_data, 
        batch_size=params["batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_data = test_dataset(
        params["meta_path"] + "test_metadata.json",
        params["data_path"], 
        (params["resize_x"], params["resize_y"]),
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

def train_tuning(params, trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders(params)

    encoder, bn = wide_resnet50_2(pretrained=True, attention=params.get("bn_attention", False))
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    proj_loss = Revisit_RDLoss(params.get("reconstruct_weight", 0.01), params.get("contrast_weight", 0.1), params.get("ssot_weight", 1.0))
    optimizer_proj = torch.optim.Adam(list(proj_layer.parameters()), lr=params.get("proj_lr", 0.001), betas=(params.get("beta1_proj", 0.5),params.get("beta2_proj", 0.999)))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=params.get("distill_lr", 0.005), betas=(params.get("beta1_distill", 0.5),params.get("beta2_distill", 0.999)))

    # lr schedulers
    distill_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_distill, 
        step_size=10,
        gamma=params["distill_lr_factor"],
    )
    proj_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_proj, 
        step_size=10,
        gamma=params["proj_lr_factor"],
    )

    best_auroc = 0
    best_epoch = 0
    num_epoch = params.get("num_epochs", 100)

    for epoch in range(1,num_epoch+1):
        bn.train()
        proj_layer.train()
        decoder.train()
        
        ## gradient acc
        accumulation_steps = 2
        
        for i, input in enumerate(train_loader):
            img = input['normal_image'].to(device)
            img_noise = input['abnormal_image'].to(device)
            inputs = encoder(img)
            inputs_noise = encoder(img_noise)

            (feature_space_noise, feature_space) = proj_layer(inputs, features_noise = inputs_noise)

            L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)

            outputs = decoder(bn(feature_space))
            L_distill = loss_function(inputs, outputs, params.get("loss_weights", [1.0, 1.0, 1.0]))
            loss = L_distill + params.get("proj_loss_weight", 0.2) * L_proj
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer_proj.step()
                optimizer_distill.step()
                # Clear gradients
                optimizer_proj.zero_grad()
                optimizer_distill.zero_grad()
        
        total_auroc, _ = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_loader, device, score_weight=params.get("score_weight"))      

        if total_auroc > best_auroc:
            best_auroc = total_auroc
            best_epoch = epoch
        
        distill_scheduler.step()
        proj_scheduler.step()

        # prune training if necessary (bad params)
        trial.report(total_auroc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_auroc

def train(params, train_loader, test_loader, device):
    encoder, bn = wide_resnet50_2(pretrained=True, attention=params.get("bn_attention", False))
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    proj_loss = Revisit_RDLoss(params.get("reconstruct_weight", 0.01), params.get("contrast_weight", 0.1), params.get("ssot_weight", 1.0))
    optimizer_proj = torch.optim.Adam(list(proj_layer.parameters()), lr=params.get("proj_lr", 0.001), betas=(params.get("beta1_proj", 0.5),params.get("beta2_proj", 0.999)))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=params.get("distill_lr", 0.005), betas=(params.get("beta1_distill", 0.5),params.get("beta2_distill", 0.999)))

    # lr schedulers
    distill_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_distill, 
        step_size=10,
        gamma=params["distill_lr_factor"],
    )
    proj_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_proj, 
        step_size=10,
        gamma=params["proj_lr_factor"],
    )

    best_auroc = 0
    best_epoch = 0
    
    auroc_dict = {}
    num_epoch = params.get("num_epochs", 100)

    print("[INFO] TRAINING MODEL...")
    for epoch in range(1,num_epoch+1):
        bn.train()
        proj_layer.train()
        decoder.train()
        loss_proj_sum = 0
        loss_distill_sum = 0
        total_loss_sum = 0
        
        ## gradient acc
        accumulation_steps = 2
        
        for i, input in enumerate(tqdm(train_loader)):
            img = input['normal_image'].to(device)
            img_noise = input['abnormal_image'].to(device)
            inputs = encoder(img)
            inputs_noise = encoder(img_noise)

            (feature_space_noise, feature_space) = proj_layer(inputs, features_noise = inputs_noise)

            L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)

            outputs = decoder(bn(feature_space))
            L_distill = loss_function(inputs, outputs, params.get("loss_weights", [1.0, 1.0, 1.0]))
            loss = L_distill + params.get("proj_loss_weight", 0.2) * L_proj
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer_proj.step()
                optimizer_distill.step()
                # Clear gradients
                optimizer_proj.zero_grad()
                optimizer_distill.zero_grad()
            
            total_loss_sum += loss.detach().cpu().item()
            loss_proj_sum += L_proj.detach().cpu().item()
            loss_distill_sum += L_distill.detach().cpu().item()
        
        avg_loss_proj = loss_proj_sum / len(train_loader)
        avg_loss_distill = loss_distill_sum / len(train_loader)
        avg_total_loss = total_loss_sum / len(train_loader)

        with open(params["log_path"], "a") as log_file:
            log_file.write("\nEPOCH {}, PROJ LOSS: {:.4f}, DISTILL LOSS:{:.4f}, TOTAL LOSS: {:.4f}".format(epoch, avg_loss_proj, avg_loss_distill, avg_total_loss))
        
        total_auroc, orchard_auroc_dict = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_loader, device, log_path=params["log_path"], score_weight=params.get("score_weight"))        
        auroc_dict[epoch+1] = orchard_auroc_dict
        print('[INFO] EPOCH {}, PROJ LOSS: {:.4f}, DISTILL LOSS:{:.4f}, TOTAL LOSS: {:.4f}, TOTAL AUROC: {:.4F}'.format(epoch, avg_loss_proj, avg_loss_distill, avg_total_loss, total_auroc))

        if total_auroc > best_auroc:
            best_auroc = total_auroc
            best_epoch = epoch

            torch.save({'proj': proj_layer.state_dict(),
                       'decoder': decoder.state_dict(),
                        'bn':bn.state_dict()}, params["model_path"])
        
        distill_scheduler.step()
        proj_scheduler.step()
    
    return best_auroc, best_epoch

# objective function for optuna
def objective(trial):
    parser = ArgumentParser(description="")
    parser.add_argument("--config", default="configs/contrast_config.yaml", required=False)
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    args = parser.parse_args()

    config = args.config
    # open config
    with open(config, "r") as ymlfile:
        params = yaml.safe_load(ymlfile)

    # objective function params
    #params["proj_lr"] = trial.suggest_float("learning_rate", low=1e-4, high=1e-1, log=True)
    #params["proj_lr_factor"] = trial.suggest_float("lr_factor", low=0.0, high=0.6)
    #params["distill_lr"] = trial.suggest_float("learning_rate", low=1e-4, high=1e-1, log=True)
    #params["distill_lr_factor"] = trial.suggest_float("lr_factor", low=0.0, high=0.6)
    #params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])
    #params["bn_attention"] = trial.suggest_categorical("bn_attention", [False, "CBAM", "SE", "GC"])
    #params["beta1_proj"] = trial.suggest_float("beta1", low=0.5, high=0.9999)
    #params["beta2_proj"] = trial.suggest_float("beta2", low=0.9, high=0.9999)
    #params["beta1_distill"] = trial.suggest_float("beta1", low=0.5, high=0.9999)
    #params["beta2_distill"] = trial.suggest_float("beta2", low=0.9, high=0.9999)

    # distill loss weights (3 levels)
    #params["loss_weight1"] = trial.suggest_float("loss_weight1", low=0.5, high=1.5)
    #params["loss_weight2"] = trial.suggest_float("loss_weight2", low=0.5, high=1.5)
    #params["loss_weight3"] = trial.suggest_float("loss_weight3", low=0.5, high=1.5)
    #params["loss_weights"] = [params["loss_weight1"], params["loss_weight2"], params["loss_weight3"]]
    #params["loss_weight_score"] = trial.suggest_categorical("loss_weight_score", [True, False])
    # score weight (max and avg of anomaly map)
    #params["score_weight"] = trial.suggest_float("score_weight", low=0.0, high=0.5)
    # weight for proj loss in total loss
    params["proj_loss_weight"] = trial.suggest_float("proj_loss_weight", low=0.0, high=1.0)
    # weight for these losses in proj loss
    params["ssot_weight"] = trial.suggest_float("ssot_weight", low=0.0, high=1.0)
    params["contrast_weight"] = trial.suggest_float("contrast_weight", low=0.0, high=1.0)
    params["reconstruct_weight"] = trial.suggest_float("reconstruct_weight", low=0.0, high=1.0)

    return train_tuning(params, trial)

if __name__ == '__main__':
    parser = ArgumentParser(description="")
    parser.add_argument("--config", default="configs/contrast_config.yaml", required=False)
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    args = parser.parse_args()

    config = args.config

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    setup_seed(111)
    if args.tune is True:
        print("[INFO] TUNING HYPERPARAMETERS...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        print("[INFO] BEST HYPERPARAMETERS:")
        trial = study.best_trial
        for key, val in trial.params.items():
            print(f"{key}: {val}")
        print(f"[INFO] BEST AUROC: {trial.value:.5f}")
    else:
        with open(config, "r") as config_file:
            params = yaml.safe_load(config_file)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print("[INFO] DEVICE:", device) 
        # create data loaders
        print("[INFO] LOADING DATA...")
        train_loader, test_loader = get_loaders(params)
            
        # train
        best_auroc, best_epoch = train(params, train_loader, test_loader, device)
        print(f"[INFO] BEST OVERALL AUROC: {best_auroc:.5f} AT EPOCH {best_epoch}")
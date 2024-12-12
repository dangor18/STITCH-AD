import time
import warnings
import random
import os
import yaml
import argparse
import optuna

import torch
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
import numpy as np

from model_utils.test_utils import evaluation, test
from model_utils.plots import plot_auroc
from model_utils.train_utils import loss_function, get_loaders, get_optimizer, create_model
from model.RD import RD

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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # evaluate every 10 epochs
        if (epoch + 1) % 2 == 0 and epoch + 1 > 3:
            total_auroc, _ = evaluation(encoder, bn, decoder, test_loader, device, score_weight=params.get("score_weight", 0.0), feature_weights=params["feature_weights"])
            
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
    model = RD(params["architecture"], params["bn_attention"], params.get("channels", 3), device, params)

    best_auroc = 0
    auroc_dict = {}
    print("[INFO] TRAINING MODEL...")
    
    # train loop    
    for epoch in range(params["num_epochs"]):
        model.train()
        loss_sum = 0.0
        num_batches = 0
        
        train_data = tqdm(train_loader)
        
        for input in train_data:
            images = input["image"].to(device)
            with autocast(device_type="cuda"):
                inputs, outputs = model(images)
                loss = loss_function(inputs, outputs, params.get("feature_weights", [1.0, 1.0, 1.0]))
            
            model.optimizer.zero_grad()

            model.scaler.scale(loss).backward()
            model.scaler.step(model.optimizer)
            model.scaler.update()
            
            loss_sum += loss.item()
            num_batches += 1

        avg_loss = loss_sum / num_batches
        
        print(f"EPOCH {epoch + 1}, LOSS: {avg_loss:.3f}")
        with open(params["log_path"], "a") as log_file:
            log_file.write(f"\nEPOCH {epoch + 1}, LOSS: {avg_loss:.3f}\n")
        
        # evaluate every 1 epoch
        if (epoch + 1) % 1 == 0:
            total_auroc, orchard_auroc_dict = evaluation(model, test_loader, device, params["log_path"], feature_weights=params["feature_weights"],
                                                         score_weight=params.get("score_weight", 0.0))
            
            # collect aurocs for each orchard and total for plotting
            auroc_dict[epoch+1] = orchard_auroc_dict
            
            print(f"EPOCH {epoch + 1}, LOSS: {avg_loss:.3f}, OVERALL AUROC: {total_auroc:.5f}")
            
            if total_auroc > best_auroc:
                best_auroc = total_auroc
                # save model
                print(f"[INFO] NEW BEST. SAVING MODEL TO {params['model_path']}...")
                torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, params["model_path"])
            
            model.scheduler.step()
    
    # after training load best model and get final metrics
    test(encoder, bn, decoder, test_loader, device, params["model_path"], score_weight=params.get("score_weight", 0.0), feature_weights=params["feature_weights"], n_plot_per_class=0)
    plot_auroc(auroc_dict)
    return best_auroc

def write_to_file(study, trial):
    """
        Write tuning output to a text file
    """
    with open("logs/optuna_results_RD.txt", "a") as f:
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

    #params["learning_rate"] = trial.suggest_float("learning_rate", low=1e-5, high=1e-2, log=True)
    #params["lr_factor"] = trial.suggest_float("lr_factor", low=0, high=0.5)
    #params["step"] = trial.suggest_int("step", low=1, high=10)
    #params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])
    #params["weight_decay"] = trial.suggest_float("weight_decay", low=1e-6, high=1e-2, log=True)
    #params["architecture"] = trial.suggest_categorical("architecture", ["wide_resnet50_2", "resnet50", "wide_resnet101_2", "asym"]) # asym for asymetric encoder decoder arch
    #params["bn_attention"] = trial.suggest_categorical("bn_attention", [False, "CBAM", "SE"])
    #params["beta1"] = trial.suggest_float("beta1", low=0.5, high=0.9999)
    #params["beta2"] = trial.suggest_float("beta2", low=0.9, high=0.9999)

    params["feature_weight1"] = trial.suggest_float("feature_weight1", low=0.5, high=1.5)
    params["feature_weight2"] = trial.suggest_float("feature_weight2", low=0.5, high=1.5)
    params["feature_weight3"] = trial.suggest_float("feature_weight3", low=0.5, high=1.5)
    params["feature_weights"] = [params["feature_weight1"], params["feature_weight2"], params["feature_weight3"]]
    params["score_weight"] = trial.suggest_float("score_weight", low=0.0, high=0.5)

    return train_tuning(params, trial)

if __name__ == '__main__':
    cwd = os.path.dirname(os.path.realpath(__file__))       # directory of the script
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", default=os.path.join(cwd, "configs/RD_config.yaml"), required=False)
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    parser.add_argument("--test", action="store_true", help="Load the model in config and test it")
    args = parser.parse_args()

    config = args.config

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    setup_seed(111)
    
    # if testing model listed in config
    if args.test is True:
        with open(config, "r") as config_file:
            params = yaml.safe_load(config_file)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("[INFO] DEVICE:", device) 
        # create data loaders
        print("[INFO] LOADING DATA...")
        train_loader, test_loader = get_loaders(params)
        encoder, bn, decoder = create_model(architecture=params["architecture"], bn_attention=params["bn_attention"], in_channels=params.get("channels", 3))
        encoder = encoder.to(device)
        bn = bn.to(device)
        encoder.eval()
        decoder = decoder.to(device)
        # test
        test(encoder, bn, decoder, test_loader, device, params["model_path"], score_weight=params.get("score_weight", 0.0), feature_weights=params["feature_weights"], n_plot_per_class=0)
        exit()
        
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
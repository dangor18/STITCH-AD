import torch
import numpy as np
import random
import os
from tqdm import tqdm
import optuna
from argparse import ArgumentParser
import yaml
from model_utils.test_utils import evaluate_RD, test_RD
from model_utils.train_utils import Revisit_RDLoss, loss_function, get_loaders_proj
from model_utils.plots import plot_auroc

from model.RevisitingRD import RevistingRD
from torch.amp import autocast, GradScaler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_tuning(params, trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_loaders_proj(params)
    model = RevistingRD(params["architecture"], params["bn_attention"], params.get("channels", 3), device, params)
    proj_loss = Revisit_RDLoss(params.get("reconstruct_weight", 0.01), params.get("contrast_weight", 0.1), params.get("ssot_weight", 1.0))

    scaler = GradScaler(device=device)
    best_auroc = 0
    best_epoch = 0
    num_epoch = params.get("num_epochs", 100)

    for epoch in range(1,num_epoch+1):
        model.train()
        ## gradient acc
        accumulation_steps = 2
        
        for i, input in enumerate(train_loader):
            img = input['normal_image'].to(device)
            img_noise = input['abnormal_image'].to(device)

            with autocast(device_type='cuda'):
                (feature_space_noise, feature_space, inputs, inputs_noise, outputs) = model(img, img_noise)   # forward pass
                # calculate proj loss and total loss
                L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)
                L_distill = loss_function(inputs, outputs, params.get("feature_weights", [1.0, 1.0, 1.0]))
                loss = L_distill + params.get("proj_loss_weight", 0.2) * L_proj
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(model.optimizer_proj)
                scaler.step(model.optimizer_distill)
                scaler.update()
                # Clear gradients
                model.optimizer_proj.zero_grad()
                model.optimizer_distill.zero_grad()
        
        # only start evaluating after 5 epochs (evaluate every epoch)
        if epoch >= 5:
            total_auroc, _ = evaluate_RD(model, test_loader, device, score_weight=params.get("score_weight"), feature_weights=params.get("feature_weights", [1.0, 1.0, 1.0]))      

            if total_auroc > best_auroc:
                best_auroc = total_auroc
            
            model.distill_scheduler.step(metrics=total_auroc)
            model.proj_scheduler.step(metrics=total_auroc)

            if epoch > 10:
                # prune training if necessary (bad params) after 10 epochs
                trial.report(total_auroc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_auroc

def train(params, train_loader, test_loader, device):
    model = RevistingRD(params["architecture"], params["bn_attention"], params.get("channels", 3), device, params)
    proj_loss = Revisit_RDLoss(params.get("reconstruct_weight", 0.01), params.get("contrast_weight", 0.1), params.get("ssot_weight", 1.0))  # projective loss function
    best_auroc = 0
    best_epoch = 0
    
    auroc_dict = {}
    num_epoch = params.get("num_epochs", 100)
    scaler = GradScaler(device=device)

    print("[INFO] TRAINING MODEL...")
    for epoch in range(1,num_epoch+1):
        model.train()
        loss_proj_sum = 0
        loss_distill_sum = 0
        total_loss_sum = 0
        
        ## gradient acc
        accumulation_steps = 2
        
        for i, input in enumerate(tqdm(train_loader, desc=f"Distill LR: {model.distill_scheduler.get_last_lr()[0]}, Proj LR: {model.proj_scheduler.get_last_lr()[0]}")):
            # input normal and psuedo-artefact image into model
            img = input['normal_image'].to(device)
            img_noise = input['abnormal_image'].to(device)

            with autocast(device_type='cuda'):
                (feature_space_noise, feature_space, inputs, inputs_noise, outputs) = model(img, img_noise)   # forward pass
                # calculate proj loss and total loss
                L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)
                L_distill = loss_function(inputs, outputs, params.get("feature_weights", [1.0, 1.0, 1.0]))
                loss = L_distill + params.get("proj_loss_weight", 0.2) * L_proj
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(model.optimizer_proj)
                scaler.step(model.optimizer_distill)
                scaler.update()
                # Clear gradients
                model.optimizer_proj.zero_grad()
                model.optimizer_distill.zero_grad()
            
            total_loss_sum += loss.detach().cpu().item()
            loss_proj_sum += L_proj.detach().cpu().item()
            loss_distill_sum += L_distill.detach().cpu().item()
        
        avg_loss_proj = loss_proj_sum / len(train_loader)
        avg_loss_distill = loss_distill_sum / len(train_loader)
        avg_total_loss = total_loss_sum / len(train_loader)

        with open(params["log_path"], "a") as log_file:
            log_file.write("\nEPOCH {}, PROJ LOSS: {:.4f}, DISTILL LOSS:{:.4f}, TOTAL LOSS: {:.4f}".format(epoch, avg_loss_proj, avg_loss_distill, avg_total_loss))
        
        # only start evaluating after 5 epochs (evaluate every epoch)
        if epoch >= 5:
            # evaluate model
            total_auroc, orchard_auroc_dict = evaluate_RD(model, test_loader, device, log_path=params["log_path"], score_weight=params.get("score_weight"), feature_weights=params.get("feature_weights", [1.0, 1.0, 1.0]))        
            auroc_dict[epoch+1] = orchard_auroc_dict
            print('[INFO] EPOCH {}, PROJ LOSS: {:.4f}, DISTILL LOSS:{:.4f}, TOTAL LOSS: {:.4f}, TOTAL AUROC: {:.4F}'.format(epoch, avg_loss_proj, avg_loss_distill, avg_total_loss, total_auroc))

            # save model if improved
            if total_auroc > best_auroc:
                best_auroc = total_auroc
                best_epoch = epoch
                print(f"[INFO] NEW BEST. SAVING MODEL TO {params['model_path']}...")
                model.save_model(params["model_path"])
            
            model.distill_scheduler.step(metrics=total_auroc)
            model.proj_scheduler.step(metrics=total_auroc)
    
    # test best model after training and plot results
    #test_RD(model, test_loader, device, model_path=params["model_path"], score_weight=params.get("score_weight"), 
    #                feature_weights=params.get("feature_weights", [1.0, 1.0, 1.0]))
    #plot_auroc(auroc_dict)
    return best_auroc, best_epoch

def write_to_file(study, trial):
    """
        Write tuning output after each trial to a text file
    """
    with open("logs/optuna_results_proj.txt", "a") as f:
        f.write(f"Trial {trial.number}:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")
        f.write("\n")

# objective function for optuna
def objective(trial, config_path):
    # open config
    with open(config_path, "r") as ymlfile:
        params = yaml.safe_load(ymlfile)

    # objective function params
    params["proj_lr"] = trial.suggest_float("proj_lr", low=1e-4, high=1e-1, log=True)
    params["distill_lr"] = trial.suggest_float("distill_lr", low=1e-4, high=1e-1, log=True)
    #params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32])
    #params["bn_attention"] = trial.suggest_categorical("bn_attention", [False, "CBAM"])
    params["beta1_proj"] = trial.suggest_categorical("beta1_proj", [0.5, 0.9])
    params["beta1_distill"] = trial.suggest_categorical("beta1_distill", [0.5, 0.9])

    params["feature_weight1"] = trial.suggest_float("feature_weight1", low=0.5, high=1.5)
    params["feature_weight2"] = trial.suggest_float("feature_weight2", low=0.5, high=1.5)
    params["feature_weight3"] = trial.suggest_float("feature_weight3", low=0.5, high=1.5)
    params["feature_weights"] = [params["feature_weight1"], params["feature_weight2"], params["feature_weight3"]]
    params["score_weight"] = trial.suggest_float("score_weight", low=0.0, high=1.0)
    
    params["proj_loss_weight"] = trial.suggest_float("proj_loss_weight", low=0.0, high=1.0)
    params["ssot_weight"] = trial.suggest_float("ssot_weight", low=0.0, high=1.0)
    params["contrast_weight"] = trial.suggest_float("contrast_weight", low=0.0, high=1.0)
    params["reconstruct_weight"] = trial.suggest_float("reconstruct_weight", low=0.0, high=1.0)

    params["octaves"] = trial.suggest_int("octaves", low=3, high=12)
    params["amplitude"] = trial.suggest_float("amplitude", low=0.5, high=1.5)
    params["frequency"] = trial.suggest_categorical("frequency", [4, 8, 16, 32, 64])
    params["persistence"] = trial.suggest_float("persistence", low=0.5, high=1.0)
    return train_tuning(params, trial)

if __name__ == '__main__':
    cwd = os.path.dirname(os.path.realpath(__file__))       # directory of the script
    parser = ArgumentParser(description="")
    parser.add_argument("--config", "-c", default=os.path.join(cwd, "configs", "RevisitingRD.yaml"), required=False)
    parser.add_argument("num_trials", type=int, nargs='?', help="Number of trials for hyperparameter tuning")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    parser.add_argument("--test", action="store_true", help="Load the model in config and test it")
    args = parser.parse_args()

    # check for the case of tuning
    if args.tune and args.num_trials is None:
        parser.error("--tune requires num_trials to be specified")

    config = os.path.join(cwd, "configs", args.config)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    setup_seed(111)
    if args.test is True:
        with open(config, "r") as config_file:
            params = yaml.safe_load(config_file)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print("[INFO] DEVICE:", device) 
        # create data loaders
        print("[INFO] LOADING DATA...")
        test_loader = get_loaders_proj(params, test=True)
            
        # test
        model = RevistingRD(params["architecture"], params["bn_attention"], params.get("channels", 3), device, params)
        test_RD(model, test_loader, device, model_path=params["model_path"], score_weight=params.get("score_weight"), 
                    feature_weights=params.get("feature_weights", [1.0, 1.0, 1.0]), n_plot_per_class=3)
        exit()
        
    if args.tune is True:
        print(f"[INFO] TUNING HYPERPARAMETERS FOR {args.num_trials} TRIALS")
        study = optuna.create_study(direction="maximize")
        objective_w_params = lambda trial: objective(trial, config)
        study.optimize(objective_w_params, n_trials=args.num_trials, callbacks=[write_to_file])
        #study.optimize(objective, n_trials=100)
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
        train_loader, test_loader = get_loaders_proj(params)
            
        # train
        best_auroc, best_epoch = train(params, train_loader, test_loader, device)
        print(f"[INFO] BEST OVERALL AUROC: {best_auroc:.5f} AT EPOCH {best_epoch}")
        with open(params["log_path"], "a") as log_file:
            log_file.write(f"************************\n")
            log_file.write(f"\nBEST OVERALL AUROC: {best_auroc:.5f} AT EPOCH {best_epoch}")
import time
from collections import defaultdict
import json
from tqdm import tqdm
import torch
import sys
import os
sys.path.insert(0, "PATCH_AD/RD/")
sys.path.insert(0, "PATCH_AD/UniAD/")
sys.path.insert(0, "PATCH_AD/SCADN/")
# RD model imports
from PATCH_AD.RD.model.RevisitingRD import RevistingRD as RevisitingRD
from PATCH_AD.RD.model.RD import RD as RD
from PATCH_AD.RD.model_utils.test_utils import cal_anomaly_score as anomaly_score_RD
from PATCH_AD.RD.model_utils.train_utils import get_loaders_proj as get_loaders_RD
# UniAD model imports
from PATCH_AD.UniAD.datasets import custom_dataset as UniAD_dataset # TODO
# SCADN model imports
from PATCH_AD.SCADN.src import custom_dataset as SCADN_dataset # TODO

def load_model(params, device):
    """
        Load the specified model from the checkpoint and return the components
        ARGS:
            params: dictionary containing parameters for the model
            device: device to run the model on
    """
    if params["model_type"] == "RD":
        return RD(params["architecture"], params["bn_attention"], params.get("channels", 3), device, params)
    elif params["model_type"] == "RevisitingRD":
        return RevisitingRD(params["architecture"], params["bn_attention"], params.get("channels", 3), device, params)
    elif params["model_type"] == "UniAD":   # TODO
        return None
    elif params["model_type"] == "SCADN":
        return None
    else:
        print("[ERROR] UNKOWN MODEL")
        return None

def get_scores_RD(params, device):
    """
        Return anomaly scores, location, and gt label for patches for each orchard in a dictionary
        ARGS:
            params: dictionary containing parameters for the model
            device: device to run the model on
    """
    # get the data loader
    data_loader = get_loaders(params)
    score_dict = defaultdict(lambda: [])
    # get the model
    model = load_model(params, device)
    # get corresponding anomaly score function
    scorer = anomaly_score_RD if params["model_type"] == "RevisitingRD" else None #TODO

    model.eval()
    with torch.no_grad():    
        for input in tqdm(data_loader):
            cls_name, lbl = input["clsname"][0], input["label"][0].item()   # cls_name either "all" or orchard id
            # libraries used prefer -1 for anomalous and 1 for normal
            if lbl == 0:
                lbl = -1

            patch = input['image'].to(device)
            x, y = input['x'].item(), input['y'].item()     # get x y location for patch from DL
            inputs, outputs = model(patch)    # get the input and output from the model

            score = scorer(inputs, outputs)    # get the anomaly score for the patch
            # TODO possibly need to wrap this in a bettter function for all the models to work
            
            score_dict[cls_name].append([x, y, score, lbl])

    return score_dict

def get_loaders(params):
    """
        Return the dataloader for inference
    """
    if params["model_type"] == "RD" or params["model_type"] == "RevisitingRD":
        data_loader = get_loaders_RD(params, test=True)
    
    return data_loader
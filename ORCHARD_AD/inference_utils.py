import time
from collections import defaultdict
import json
from tqdm import tqdm
import torch
import sys
import os
from easydict import EasyDict
import yaml
from mmcv import Config
import cv2
import torch.nn.functional as F

sys.path.insert(0, "PATCH_AD/RD/")
sys.path.insert(0, "PATCH_AD/UniAD/")
sys.path.insert(0, "PATCH_AD/SCADN/")
# RD model imports
from PATCH_AD.RD.model.RevisitingRD import RevistingRD as RevisitingRD
from PATCH_AD.RD.model.RD import RD as RD
from PATCH_AD.RD.model_utils.test_utils import cal_anomaly_score as anomaly_score_RD
from PATCH_AD.RD.model_utils.train_utils import get_loaders_proj as get_loaders_RD
# UniAD model imports
from PATCH_AD.UniAD.models.model_helper import ModelHelper as UniAD_model
from PATCH_AD.UniAD.utils.misc_helper import update_config
from PATCH_AD.UniAD.datasets.data_builder import build_dataloader
from PATCH_AD.UniAD.run_inference import encode_pred as encode_pred_UniAD
# SCADN model imports
from PATCH_AD.SCADN.src import custom_dataset as SCADN_dataset # TODO
from PATCH_AD.SCADN.src.custom_experiments import ExpStitchO

import numpy as np

from scipy.ndimage import laplace

def load_config_UniAD(config_path):
    """
    Load and update the configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        EasyDict: Updated configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    return update_config(config)

def load_config_SCADN(config_path):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """
    # load config file
    config = Config.fromfile(config_path)

    if config.SUB_SET is not None:
        config.SUB_SET = str(config.SUB_SET)
        config.PATH = os.path.join(config.PATH, config.SUB_SET)

    # create checkpoints path if does't exist
    if not os.path.exists(config.PATH):
        os.makedirs(config.PATH)

    return config

def load_model(model_type, params, device):
    """
        Load the specified model from the checkpoint and return the components
        ARGS:
            params: dictionary containing parameters for the model
            device: device to run the model on
    """
    if model_type == "RD":
        model = RD(params["architecture"], params["bn_attention"], params.get("channels", 3), device, params)
        model.load_model(params["model_checkpoint"])
        model.eval()
        return model
    elif model_type == "RevisitingRD":
        model = RevisitingRD(params["architecture"], params["bn_attention"], params.get("channels", 3), device, params)
        model.load_model(params["model_checkpoint"])
        model.eval()
        return model
    elif model_type == "UniAD":
        model = UniAD_model(params.net)
        model.cuda()
        checkpoint = torch.load(params.saver.load_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        thresholds = checkpoint.get('thresholds', {})   # TODO
        model.eval()
        if 'all_case_1' not in thresholds or 'all_case_2' not in thresholds:
            raise ValueError("Checkpoint does not contain required thresholds for all_case_1 and all_case_2")
        return model
    elif model_type == "SCADN":
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in params.GPU)
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

        # init device
        if torch.cuda.is_available():
            params.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
            params.DEVICE = torch.device("cpu")

        # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
        cv2.setNumThreads(0)

        model = ExpStitchO(params)
        model.load()
        return model
    else:
        print("[ERROR] UNKOWN MODEL")
        return None

def get_scores_UniAD(model, data_loader, params, device):
    # get the data loader
    score_dict = defaultdict(lambda: [])
    # get corresponding anomaly score function

    model.eval()
    with torch.no_grad():    
        for input in tqdm(data_loader):
            cls_name, lbl = input["clsname"][0], input["label"][0].item()   # cls_name either "all" or orchard id
            # libraries used prefer -1 for anomalous and 1 for normal
            if lbl == 0:
                lbl = 1
            elif lbl == 1:
                lbl = -1

            x, y = input['x'].item(), input['y'].item()     # get x y location for patch from DL

            outputs = model(input)    # get the input and output from the model
            preds = outputs["pred"].cpu().numpy()
            score = float(encode_pred_UniAD(preds)[0])
            
            score_dict[cls_name].append([x, y, score, lbl])

    return score_dict

def get_scores_SCADN(model, data_loader, params, device):
    score_dict = model.get_score_dict()
    # min max scale the scores for each orchard
    for orchard in score_dict:
        scores = [score for _, _, score, _ in score_dict[orchard]]
        min_score, max_score = min(scores), max(scores)
        for i in range(len(score_dict[orchard])):
            x, y, score, lbl = score_dict[orchard][i]
            score_dict[orchard][i] = [x, y, (score - min_score) / (max_score - min_score), lbl]
    return score_dict

def get_scores_RD(model, data_loader, params, device):
    """
        Return anomaly scores, location, and gt label for patches for each orchard in a dictionary
        ARGS:
            model: RD model to use for inference
            params: dictionary containing parameters for the model
            device: device to run the model on
    """
    # get the data loader
    score_dict = defaultdict(lambda: [])
    # get corresponding anomaly score function

    model.eval()
    with torch.no_grad():    
        for input in tqdm(data_loader):
            cls_name, lbl = input["clsname"][0], input["label"][0].item()   # cls_name either "all" or orchard id
            # libraries used prefer -1 for anomalous and 1 for normal
            #print(lbl)
            if lbl == 0:
                lbl = 1
            elif lbl == 1:
                lbl = -1

            patch = input['image'].to(device)
            x, y = input['x'].item(), input['y'].item()     # get x y location for patch from DL
            inputs, outputs = model(patch)    # get the input and output from the model

            score = anomaly_score_RD(inputs, outputs, score_weight=params["score_weight"], feature_weights=params["feature_weights"])    # get the anomaly score for the patch
            DEM = patch[0, 0].cpu().numpy()
            # get std
            std_dem = np.std(DEM).item()  # std of DEM channel
            # get mean
            mean = np.mean(DEM).item() # mean of DEM channel
            complexity = np.sum(np.abs(laplace(DEM))).item()

            score_dict[cls_name].append([x, y, score, mean, std_dem, complexity, lbl])

    return score_dict

def remap_coordinates(score_dict, sort_by_new=False):
    """
    Remaps the coordinates in score_dict to start from (0,0) and increment by 1.
    This handles "jumps" in the original coordinate system caused by masked regions.
    
    Args:
        score_dict: Dictionary mapping orchard names to lists of [x, y, score, mean, std_dem, complexity, lbl]
        sort_by_new: If True, sorts the entries by new_y, then new_x (optional)
    
    Returns:
        Updated score_dict with remapped coordinates
    """
    remapped_dict = defaultdict(list)
    
    for cls_name, entries in score_dict.items():
        # Create a mapping from original coordinates to new indices
        mapping = {}
        
        # Extract unique (x, y) pairs
        unique_xy = set((entry[0], entry[1]) for entry in entries)
        
        # Sort by y, then by x
        sorted_xy = sorted(unique_xy, key=lambda xy: (xy[1], xy[0]))
        
        current_y = None
        new_y = -1
        new_x = 0
        
        for x, y in sorted_xy:
            if y != current_y:
                # New row
                current_y = y
                new_y += 1
                new_x = 0
            
            mapping[(x, y)] = (new_x, new_y)
            new_x += 1
        
        # Apply the mapping to all entries
        remapped_entries = []
        for entry in entries:
            x, y, score, mean, std_dem, complexity, lbl = entry
            new_x, new_y = mapping[(x, y)]
            remapped_entries.append([new_x, new_y, score, mean, std_dem, complexity, lbl])
        
        # Sort by new coordinates if requested
        if sort_by_new:
            remapped_entries.sort(key=lambda e: (e[1], e[0]))
        
        remapped_dict[cls_name] = remapped_entries
    
    return remapped_dict

def get_loaders(model_type, params):
    """
        Return the dataloader for inference
    """
    if model_type == "RD" or model_type == "RevisitingRD":
        data_loader = get_loaders_RD(params, test=True)
    elif model_type == "UniAD":
        data_loader = build_dataloader(params.dataset, distributed=False, inference=True)
    elif model_type == "SCADN":
        data_loader = None
    return data_loader
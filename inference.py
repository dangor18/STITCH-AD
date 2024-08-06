import numpy as np
import torch
from torch.utils.data import DataLoader
from model_utils.test_utils import cal_anomaly_map, gaussian_filter
from model.resnet import wide_resnet50_2
from model_utils.train_utils import MultiProjectionLayer
from model.de_resnet import de_wide_resnet50_2
from argparse import ArgumentParser
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import zscore
import random
import matplotlib.pyplot as plt
from data.DL_inference import inference_dataset

gt_dict = {
    "1676": -1,
    "1883": 1,
    "1996": -1,
    "2057": -1,
    "1724": -1,
    "2222": 1,
    "2227": 1,
    "2228": 1,
    "2240": 1,
    "2848": 1,
}

def load_model(model_type: str, model_path: str, in_channels: int, device):
    """
        Load the necessary model from the checkpoint and return the components
        ARGS:
            model_type: RD or RDProj
            model_path: path to the model checkpoint
            in_channels: number of input channels (1 or 3)
            device: device to run the model on
    """
    encoder, bn = wide_resnet50_2(pretrained=True, attention=params["bn_attention"], in_channels=in_channels)
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    encoder.eval()
    bn.eval()
    decoder.eval()
    if model_type == "RD":
        ckp = torch.load(model_path)
        for k, v in list(ckp['bn'].items()):
            if 'memory' in k:
                ckp['bn'].pop(k)
        decoder.load_state_dict(ckp['decoder'])
        bn.load_state_dict(ckp['bn'])
        
        return encoder, bn, decoder
    
    elif model_type == "RDProj":
        proj_layer = MultiProjectionLayer(base=64).to(device)
        proj_layer.eval()
        ckp = torch.load(model_path)
        for k, v in list(ckp['bn'].items()):
            if 'memory' in k:
                ckp['bn'].pop(k)
        decoder.load_state_dict(ckp['decoder'])
        bn.load_state_dict(ckp['bn'])
        proj_layer.load_state_dict(ckp['proj'])

        return encoder, bn, decoder, proj_layer
    else:
        print("[ERROR] UNKOWN MODEL")
        return None

def get_scores(params, data_loader, device):
    """
        Return anomaly scores for patches for each orchard
        ARGS:
            params: dictionary containing parameters for the model
            data_loader
            device: device to run the model on
    """
    score_dict = {}
    if params["model_type"] == "RD":
        encoder, bn, decoder = load_model("RD", params["model_path"], params["channels"], device)
    else:
        encoder, bn, decoder, proj_layer = load_model("RDProj", params["model_path"], params["channels"], device)
    
    for input in data_loader:
        patch = input['image'].to(device)
        orchard_id = input['clsname'][0]
        if score_dict.get(orchard_id) is None:  # new orchard
            score_dict[orchard_id] = []
        with torch.no_grad():
            if params["model_type"] == "RD":    # get score from model and add to list
                inputs = encoder(patch)
                outputs = decoder(bn(inputs))
                anomaly_map, _ = cal_anomaly_map(inputs, outputs, patch.shape[-1], amap_mode='a', weights=params["loss_weights"])
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                score = np.max(anomaly_map) + params.get("score_weight", 0) * np.mean(anomaly_map)
                score_dict[orchard_id].append(score)
            elif params["model_type"] == "RDProj":
                inputs = encoder(patch)
                features = proj_layer(inputs)
                outputs = decoder(bn(features))
                anomaly_map, _ = cal_anomaly_map(inputs, outputs, patch.shape[-1], amap_mode='a', weights=params["loss_weights"])
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                score = np.max(anomaly_map) + params.get("score_weight", 0) * np.mean(anomaly_map)
                score_dict[orchard_id].append(score)
    
    return score_dict

def get_loaders(params):
    """
        Return the dataloader for inference
    """
    dataset = inference_dataset(
        params["meta_path"] + "train_metadata.json", 
        params["meta_path"] + "test_metadata.json",
        params["data_path"], 
        (params["resize_x"], params["resize_y"]), 
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    return data_loader

def normalize(patches):
    '''
        Normalize patches to lie between -1 and 1
    '''
    patches = (patches - np.min(patches)) / (np.max(patches) - np.min(patches))
    patches = (patches * 2) - 1
    return patches

def divide_scores(scores, min_size):
    '''
        Divide the scores into patches of size min_size. If the size of the scores is not a multiple of min_size, the last patch will contain random scores from the scores list
    '''
    patches = []
    num_segments = len(scores) // min_size
    diff = len(scores) - (num_segments * min_size)
    random_scores = np.random.choice(scores, diff)

    for i in range(num_segments-1):
        patches.append(scores[i*min_size:(i+1)*min_size])
    # add the last section with the random scores
    patches.append(np.concatenate((scores[(num_segments-1)*min_size:], random_scores)))

    return patches

def infer_iso_forest(params, data_loader, device):
    '''
        Perform orchard level inference with isolation forest
        ARGS:
            model_type: model to be used for inference (RD, RDProj)
            data_loader
            device: device to run the model on
    '''
    score_dict = get_scores(params, data_loader, device)
    pr_dict = {}
    anomalous_outliers = 0
    normal_outliers = 0
    
    # initialize isolation forest
    clf = IsolationForest(contamination=params["contamination"], random_state=42)
    
    for orchard_id, scores in score_dict.items():
        clf.fit(np.array(scores).reshape(-1, 1))    # fit the isolation forest
        pr_dict[orchard_id] = clf.predict(np.array(scores).reshape(-1, 1))
        print(orchard_id + "\n", np.unique(pr_dict[orchard_id], return_counts=True))
        # if the number of anomalies is greater than the threshold, classify as anomalous
        if np.unique(pr_dict[orchard_id], return_counts=True)[1][0] > params["forest_threshold"]:
            # if classified as anomalous tally the number of patches for normal and anomalous orchards (used in tuning)
            if gt_dict[orchard_id] == 1:
                anomalous_outliers += np.unique(pr_dict[orchard_id], return_counts=True)[1][0]
            else:
                normal_outliers += np.unique(pr_dict[orchard_id], return_counts=True)[1][0]
            pr_dict[orchard_id] = -1
        else:
            pr_dict[orchard_id] = 1
        
    return pr_dict, anomalous_outliers, normal_outliers

def infer_zstat(params, data_loader, device):
    '''
        Perform orchard level inference using Z statistics where the threshold is n units of std dev higher or lower than the mean 
        ARGS:
            model_type: model to be used for inference (RD, RDProj)
            data_loader
            device: device to run the model on
    '''
    score_dict = get_scores(params, data_loader, device)
    pr_dict = {}
    # tally the outlier patches for anomalous and normal orchards
    anomalous_outliers = 0
    normal_outliers = 0
    
    for orchard_id, scores in score_dict.items():
        z_scores = np.abs(zscore(scores))   # z score normalization for all patches in an orchard
        outliers = z_scores > params["threshold"] + z_scores < -params["threshold"]
        if np.any(outliers):  # if there are any outliers
            # if classified as anomalous tally the number of patches for normal and anomalous orchards (used in tuning)
            if gt_dict[orchard_id] == 1:
                anomalous_outliers += len(outliers)
            else:
                normal_outliers += len(outliers)
            pr_dict[orchard_id] = -1
        else:
            pr_dict[orchard_id] = 1
    
    return pr_dict, anomalous_outliers, normal_outliers

def infer_dbscan(params, data_loader, device):
    '''
        Perform orchard level inference using DBSCAN clustering
        ARGS:
            model_type: model to be used for inference (RD, RDProj)
            data_loader
            device: device to run the model on
    '''
    score_dict = get_scores(params, data_loader, device)
    pr_dict = {}
    
    dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])

    for orchard_id, scores in score_dict.items():
        dbscan.fit(np.array(scores).reshape(-1, 1))     # fit the model 
        print(orchard_id + "\n", np.unique(dbscan.labels_, return_counts=True))
        pr_dict[orchard_id] = dbscan.labels_            # get predictions
        if np.unique(pr_dict[orchard_id], return_counts=True)[1][0] > params["min_score"]:
            pr_dict[orchard_id] = -1
        else:
            pr_dict[orchard_id] = 1
        
    return pr_dict

def dbscan_scorer(labels):
    """
        Score the DBSCAN clustering based on the number of clusters, noise points, and cluster sizes
    """
    num_clusters = len(np.unique(labels, return_counts=True))
    num_noise = list(labels).count(-1)
    total_points = len(labels)
    
    # score for number of clusters
    if num_clusters == 1:
        anomaly_score = 0
    elif num_clusters == 2:
        anomaly_score = 10
    else:
        anomaly_score = 50
    
    # add ratio of noise to total
    noise_ratio = num_noise / total_points
    anomaly_score += noise_ratio * 100
    
    # adjust for cluster size variance
    if num_clusters > 1:
        cluster_sizes = [list(labels).count(i) for i in range(max(labels) + 1)]
        size_variance = max(cluster_sizes) / min(cluster_sizes)
        anomaly_score += min(size_variance * 10, 25)
    
    return min(anomaly_score, 100)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", type=str, default="configs/inference.yaml")
    args = arg_parser.parse_args()
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = get_loaders(params)
    if params["method"] == "Z":
        pr_dict = infer_zstat(params, data_loader, device)
    elif params["method"] == "ISO_FOREST":
        pr_dict = infer_iso_forest(params, data_loader, device)
    elif params["method"] == "DBSCAN":
        pr_dict = infer_dbscan(params, data_loader, device)
    else:
        print("[ERROR] UNKNOWN METHOD")
    

    #print(pr_dict)

    gt = [gt_dict[orchard_id] for orchard_id in pr_dict.keys()]
    pr = list(pr_dict.values())
    cm = confusion_matrix(gt, pr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalous"])
    disp.plot()
    plt.show()
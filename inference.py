import numpy as np
import torch
from torch.utils.data import DataLoader
from model_utils.test_utils import cal_anomaly_map
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

def get_loaders(params):
    dataset = inference_dataset(
        params["meta_path"] + "train_metadata.json", 
        params["meta_path"] + "test_metadata.json",
        params["data_path"], 
        (params["resize_x"], params["resize_y"]), 
    )

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    return data_loader

def load_model(model_type: str, model_path: str, in_channels: int, device):
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
            model_type: model to be used for inference (Baseline, Projective RD)
            test_loader: dataloader for test data (normal patches)
            train_loader: dataloader for train data (normal and anomalous patches)
            device: device to run the model on
    '''
    score_dict = {}
    pr_dict = {}
    fit_scores = []     # scores to fit the isolation forest
    if params["model_type"] == "RD":
        encoder, bn, decoder = load_model("RD", params["model_path"], params["channels"], device)
    else:
        encoder, bn, decoder, proj_layer = load_model("RDProj", params["model_path"], params["channels"], device)
    
    # initialize isolation forest
    clf = IsolationForest(contamination=params["contamination"], random_state=42)

    for input in data_loader:
        patch = input['image'].to(device)
        orchard_id = input['clsname'][0]
        if score_dict.get(orchard_id) is None:  # new orchard
            score_dict[orchard_id] = []
        with torch.no_grad():
            if params["model_type"] == "RD":    # get score from model and add to list
                inputs = encoder(patch)
                outputs = decoder(bn(inputs))
                score = cal_anomaly_map(inputs, outputs, amap_mode='a', weights=params["loss_weights"])
                score_dict[orchard_id].append(score.cpu())
            elif params["model_type"] == "RDProj":
                inputs = encoder(patch)
                features = proj_layer(inputs)
                outputs = decoder(bn(features))
                score = cal_anomaly_map(inputs, outputs, amap_mode='a', weights=params["loss_weights"])
                score_dict[orchard_id].append(score.cpu())
    
    #min_size = min(len(value) for value in score_dict.values())
    for orchard_id, scores in score_dict.items():
        #scores = normalize(np.array(scores))        # normalize
        #scores = divide_scores(scores, min_size)    # segment scores
        #sample = random.sample(scores, len(scores) // 4)  # take a quarter of the scores for each orchard to fit the isolation forest
        #for s in sample:
            #fit_scores.append(s)
        clf.fit(np.array(scores).reshape(-1, 1))    # fit the isolation forest
        pr_dict[orchard_id] = clf.predict(np.array(scores).reshape(-1, 1))
    #clf.fit(np.array(fit_scores).reshape(-1, 1))    # fit the isolation forest
    # now predict on each segment per orchard and get the final prediction
    #for orchard_id, scores in score_dict.items():
        #for s in scores:
            #pr = clf.predict(s.reshape(-1, 1))
            #pr_dict[orchard_id] = 1 if -1 in pr else 0
        
    return pr_dict

def infer_zstat(params, data_loader, device):
    '''
        Perform orchard level inference using IQR
        ARGS:
            model_type: model to be used for inference (Baseline, Projective RD)
            test_loader: dataloader for test data (normal patches)
            train_loader: dataloader for train data (normal and anomalous patches)
            device: device to run the model on
    '''
    score_dict = {}
    pr_dict = {}
    if params["model_type"] == "RD":
        encoder, bn, decoder = load_model("RD", params["model_path"], params["channels"], device)
    else:
        encoder, bn, decoder, proj_layer = load_model("RDProj", params["model_path"], params["channels"], device)

    for input in data_loader:
        patch = input['image'].to(device)
        orchard_id = input['clsname'][0]
        if score_dict.get(orchard_id) is None:
            score_dict[orchard_id] = []
        with torch.no_grad():
            if params["model_type"] == "RD":
                inputs = encoder(patch)
                outputs = decoder(bn(inputs))
                score = cal_anomaly_map(inputs, outputs, amap_mode='a', weights=params["loss_weights"])
                score_dict[orchard_id].append(score.cpu())
            elif params["model_type"] == "RDProj":
                inputs = encoder(patch)
                features = proj_layer(inputs)
                outputs = decoder(bn(features))
                score = cal_anomaly_map(inputs, outputs, amap_mode='a', weights=params["loss_weights"])
                score_dict[orchard_id].append(score.cpu())
    
    #min_size = min(len(value) for value in score_dict.values())
    for orchard_id, scores in score_dict.items():
        z_scores = np.abs(zscore(scores))   # score normalization per orchard
        if len(scores[z_scores > params["threshold"]]) > 0:    # if there are outliers
            pr_dict[orchard_id] = 1
        else:
            pr_dict[orchard_id] = -1
    
    return pr_dict

def infer_dbscan(params, data_loader, device):
    '''
        Perform orchard level inference using KMeans Clustering
        ARGS:
            model_type: model to be used for inference (Baseline, Projective RD)
            test_loader: dataloader for test data (normal patches)
            train_loader: dataloader for train data (normal and anomalous patches)
            device: device to run the model on
    '''
    score_dict = {}
    pr_dict = {}
    fit_scores = []     # scores to fit the isolation forest
    if params["model_type"] == "RD":
        encoder, bn, decoder = load_model("RD", params["model_path"], params["channels"], device)
    else:
        encoder, bn, decoder, proj_layer = load_model("RDProj", params["model_path"], params["channels"], device)
    
    dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])

    for input in data_loader:
        patch = input['image'].to(device)
        orchard_id = input['clsname'][0]
        if score_dict.get(orchard_id) is None:
            score_dict[orchard_id] = []
        with torch.no_grad():
            if params["model_type"] == "RD":
                inputs = encoder(patch)
                outputs = decoder(bn(inputs))
                score = cal_anomaly_map(inputs, outputs, amap_mode='a', weights=params["loss_weights"])
                score_dict[orchard_id].append(score.cpu())
            elif params["model_type"] == "RDProj":
                inputs = encoder(patch)
                features = proj_layer(inputs)
                outputs = decoder(bn(features))
                score = cal_anomaly_map(inputs, outputs, amap_mode='a', weights=params["loss_weights"])
                score_dict[orchard_id].append(score.cpu())
    
    #min_size = min(len(value) for value in score_dict.values())
    for orchard_id, scores in score_dict.items():
        #scores = normalize(np.array(scores))        # normalize
        #scores = divide_scores(scores, min_size)    # segment scores
        #sample = random.sample(scores, len(scores) // 4)  # take a quarter of the scores for each orchard to fit the isolation forest
        #for s in sample:
            #fit_scores.append(s)
        
        dbscan.fit(np.array(fit_scores).reshape(-1, 1))     # fit the kmeans
        dbscan.labels_ = np.array(scores).reshape(-1, 1)    # get predictions
        pr_dict[orchard_id] = dbscan.labels_
        
    return pr_dict

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
    
    gt = [gt_dict[orchard_id] for orchard_id in pr_dict.keys()]
    pr = list(pr_dict.values())
    cm = confusion_matrix(gt, pr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalous"])
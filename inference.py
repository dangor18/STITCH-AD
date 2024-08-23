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
from hdbscan import HDBSCAN
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data.DL_inference import inference_dataset
import time
from tqdm import tqdm
import json
from sklearn.preprocessing import StandardScaler

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

def load_model(params, device):
    """
        Load the specified model from the checkpoint and return the components
        ARGS:
            params: dictionary containing parameters for the model
            device: device to run the model on
    """
    encoder, bn = wide_resnet50_2(pretrained=True, attention=params["bn_attention"], in_channels=params["channels"])
    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False).to(device)
    if params["model_type"] == "RD":
        ckp = torch.load(params["model_path"], weights_only=True)
        for k, v in list(ckp['bn'].items()):
            if 'memory' in k:
                ckp['bn'].pop(k)
        decoder.load_state_dict(ckp['decoder'])
        bn.load_state_dict(ckp['bn'])

        encoder.eval()
        bn.eval()
        decoder.eval()
        
        return encoder, bn, decoder
    
    elif params["model_type"] == "RDProj":
        proj_layer = MultiProjectionLayer(base=64).to(device)
        ckp = torch.load(params["model_path"], weights_only=True)
        for k, v in list(ckp['bn'].items()):
            if 'memory' in k:
                ckp['bn'].pop(k)
        decoder.load_state_dict(ckp['decoder'])
        bn.load_state_dict(ckp['bn'])
        proj_layer.load_state_dict(ckp['proj'])
            
        encoder.eval()
        bn.eval()
        decoder.eval()
        proj_layer.eval()

        return encoder, bn, decoder, proj_layer
    else:
        print("[ERROR] UNKOWN MODEL")
        return None

def get_scores_loc(params, data_loader, device):
    """
        Return anomaly scores, location, and gt label for patches for each orchard
        ARGS:
            params: dictionary containing parameters for the model
            data_loader
            device: device to run the model on
    """
    start_time = time.time()
    print("GETTING PATCH DATA...")
    score_dict = {}
    if params["model_type"] == "RD":
        encoder, bn, decoder = load_model(params, device)
    else:
        encoder, bn, decoder, proj_layer = load_model(params, device)
    
    for input in tqdm(data_loader):
        patch = input['image'].to(device)
        x, y = input['x'].item(), input['y'].item() # get x y location for patch from DL
        lbl = input['label'].item()
        if lbl in [1, 2, 3]:    # convert gt labels to binary labels
            lbl = -1
        elif lbl == 0:
            lbl = 1
        orchard_id = input['clsname'][0]
        
        if orchard_id not in score_dict:
            score_dict[orchard_id] = []

        with torch.no_grad():
            if params["model_type"] == "RD":    # get score from model and add to list
                inputs = encoder(patch)
                outputs = decoder(bn(inputs))
            elif params["model_type"] == "RDProj":
                inputs = encoder(patch)
                features = proj_layer(inputs)
                outputs = decoder(bn(features))
        anomaly_map, _ = cal_anomaly_map(inputs, outputs, patch.shape[-1], amap_mode='a', weights=params["feature_weights"])
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        score = np.max(anomaly_map) + params.get("score_weight", 0) * np.average(anomaly_map)
        
        score_dict[orchard_id].append([x, y, score, lbl])

    end_time = time.time()
    run_time = end_time - start_time
    print("TIME (s):", run_time)
    # write dict to file (only used for the demo)
    with open(f"data/{params['model_type']}_score_dict.json", "w") as f:
        json.dump(score_dict, f)

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

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    return data_loader
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

def infer_iso_forest(params, score_dict):
    '''
        Perform orchard level inference with isolation forest
        ARGS:
            params: config dictionary
            score_dict: dictionary containing the scores for each patch for each orchard
    '''
    pr_dict = {}
    # initialize confusion matrices
    normal_cm = np.zeros((2, 2))
    anomalous_cm = np.zeros((2, 2))
    
    # initialize isolation forest
    clf = IsolationForest(contamination=params["contamination"], random_state=42)
    print("NUMBER OF OUTLIERS TO NORMAL PATCHES DETECTED FOR EACH ORCHARD:")
    #clf.fit(np.concatenate([scores for scores in score_dict.values()]).reshape(-1, 1))    # fit the isolation forest
    for orchard_id, data in score_dict.items():
        scores = np.array([item[2] for item in data])           # get each patches score
        locations = np.array([item[0:2] for item in data])      # get each patches location
        gt_labels = np.array([item[3] for item in data])        # get each patches label (used for evaluation only)
        #print(orchard_id, np.unique(gt_labels, return_counts=True))
        
        # combine scores and locations
        features = np.column_stack((scores, locations))
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        clf.fit(features_normalized)
        predictions = clf.predict(features_normalized)
        print(orchard_id + "\n", np.unique(predictions, return_counts=True))

        FP = np.sum((predictions == -1) & (gt_labels == 1))     # false positives
        FN = np.sum((predictions == 1) & (gt_labels == -1))     # false negatives
        TP = np.sum((predictions == -1) & (gt_labels == -1))    # true positives
        TN = np.sum((predictions == 1) & (gt_labels == 1))      # true negatives

        if gt_dict[orchard_id] == 1:  # normal orchard
            normal_cm += np.array([[TN, FP], [FN, TP]])
        else:  # anomalous orchard
            anomalous_cm += np.array([[TN, FP], [FN, TP]])

        # if the number of anomalies is greater than the threshold, classify as anomalous
        if np.unique(predictions, return_counts=True)[1][0] > params["forest_threshold"]:
            pr_dict[orchard_id] = -1
        else:
            pr_dict[orchard_id] = 1
        
    return pr_dict, normal_cm, anomalous_cm

def infer_dbscan(params, score_dict):
    '''
        Perform orchard level inference using DBSCAN clustering
        ARGS:
            params: config dictionary
            score_dict: dictionary containing the scores for each patch for each orchard
    '''
    pr_dict = {}
    # initialize confusion matrices
    normal_cm = np.zeros((2, 2))
    anomalous_cm = np.zeros((2, 2))

    hdbscan = HDBSCAN(min_cluster_size=5, min_samples=5, cluster_selection_epsilon=0.1, alpha=1.0)
    for orchard_id, data in score_dict.items():
        scores = np.array([item[2] for item in data])           # get each patches score
        locations = np.array([item[0:2] for item in data])      # get each patches location
        gt_labels = np.array([item[3] for item in data])        # get each patches ground truth label (Anom or Normal) (used for evaluation only)
        #print(orchard_id, np.unique(gt_labels, return_counts=True))

        features = np.column_stack((locations, scores))
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        cluster_labels = hdbscan.fit_predict(features_normalized)     # fit the model
       
        #print(orchard_id + "\n", np.unique(hdbscan.labels_, return_counts=True))
        # find the largest cluster (ignore noise / -1)
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
        if len(unique_labels) == 0:
            pr_dict[orchard_id] = 1
            continue
        else:
            normal_label = unique_labels[np.argmax(counts)]

            # calculate the mean score (z component) for each cluster and also its size
            cluster_stats = []
            normal_mean = np.mean(features_normalized[cluster_labels == normal_label, 2])
            for label in unique_labels:
                if label != -1:
                    cluster_data = features_normalized[cluster_labels == label]
                    cluster_stats.append([label, np.mean(cluster_data[:, 2]), len(cluster_data)])
            #print(orchard_id, cluster_stats)
            
            # identify whether anomalous clusters exist (those with enough elements and large mean score)
            anom_clusters = []
            norm_clusters = [normal_label, -1]
            pr_dict[orchard_id] = 1
            for label, mean_z, size in cluster_stats:
                if label != normal_label:
                    if (mean_z > normal_mean + params["v_thresh"] and
                        size >= params["min_cluster_size"]):
                        pr_dict[orchard_id] = -1
                        anom_clusters.append(label)
                    else:
                        norm_clusters.append(label)
        #print(orchard_id, anom_clusters, norm_clusters)
        
        predictions = np.zeros(cluster_labels.shape)
        predictions[np.isin(cluster_labels, anom_clusters)] = -1
        predictions[np.isin(cluster_labels, norm_clusters)] = 1

        FP = np.sum((predictions == -1) & (gt_labels == 1))     # false positives
        FN = np.sum((predictions == 1) & (gt_labels == -1))     # false negatives
        TP = np.sum((predictions == -1) & (gt_labels == -1))    # true positives
        TN = np.sum((predictions == 1) & (gt_labels == 1))      # true negatives

        if gt_dict[orchard_id] == 1:  # normal orchard
            normal_cm += np.array([[TN, FP], [FN, TP]])
        else:  # anomalous orchard
            anomalous_cm += np.array([[TN, FP], [FN, TP]])
        
        # plot the clusters
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(features_normalized[:, 0], features_normalized[:, 1], features_normalized[:, 2], c=cluster_labels, cmap='hsv')
        fig.colorbar(scatter)
        
        ax.set_title(f'3D Adaptive HDBSCAN Clustering for Orchard {orchard_id}')
        ax.set_xlabel('X Location')
        ax.set_ylabel('Y Location')
        ax.set_zlabel('Score')
        plt.show()
        
    return pr_dict, normal_cm, anomalous_cm

def print_results(pr_dict, normal_cm, anomalous_cm):
    """
        Print the results of the inference
        ARGS:
            pr_dict: dictionary containing the predictions for each orchard
            normal_cm: confusion matrix for normal orchards
            anomalous_cm: confusion matrix for anomalous orchards
    """
    for orchard_id, pr in pr_dict.items():
        print(f"\nORCHARD ID: {orchard_id}    PREDICTION: {pred_list[pr]}     GROUND TRUTH: {pred_list[gt_dict[orchard_id]]}")
    
    print("ORCHARD CLASSIFICATION ACCURACY:", np.sum([1 for k, v in pr_dict.items() if v == gt_dict[k]]) / len(pr_dict))
    disp_anom = ConfusionMatrixDisplay(anomalous_cm, display_labels=["Normal", "Anomalous"])
    f1_anom = 2 * anomalous_cm[1, 1] / (2 * anomalous_cm[1, 1] + anomalous_cm[0, 1] + anomalous_cm[1, 0])
    disp_norm = ConfusionMatrixDisplay(normal_cm, display_labels=["Normal", "Anomalous"])
    f1_norm = 2 * normal_cm[0, 0] / (2 * normal_cm[0, 0] + normal_cm[1, 0] + normal_cm[0, 1])
    disp_anom.plot(values_format='', cmap='Blues')
    plt.title("Anomalous Orchards Confusion Matrix")
    plt.show()
    disp_norm.plot(values_format='', cmap='Blues')
    plt.title("Normal Orchards Confusion Matrix")
    plt.show()
    print("ANOMALOUS ORCHARD F1 SCORE:", f1_anom)
    print("NORMAL ORCHARD F1 SCORE:", f1_norm)

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", type=str, default="configs/inference.yaml")
    arg_parser.add_argument("--demo", action="store_true", help="load stored data from DL model instead of infering on each patch. Just here for making the demo faster")
    args = arg_parser.parse_args()
    with open(args.config, "r") as f:
        params = yaml.safe_load(f)

    pred_list = {-1: "Anomalous", 1: "Normal"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #plot_score_grid(get_scores_loc(params, get_loaders(params), device))
    if args.demo:
        with open(f"data/{params['model_type']}_score_dict.json", "r") as f:
            score_dict = json.load(f)
    else:
        score_dict = get_scores_loc(params, get_loaders(params), device)

    pr_dict, normal_cm, anomalous_cm = infer_iso_forest(params, score_dict)
    print("===================== ISOLATION FOREST =====================")
    print_results(pr_dict, normal_cm, anomalous_cm)
    pr_dict, normal_cm, anomalous_cm = infer_dbscan(params, score_dict)
    print("========================== DBSCAN ==========================")
    print_results(pr_dict, normal_cm, anomalous_cm)
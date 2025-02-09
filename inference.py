import numpy as np
import torch
import sys
import os
from argparse import ArgumentParser
import yaml
from sklearn.ensemble import IsolationForest
from hdbscan import HDBSCAN
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import json
import optuna
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from ORCHARD_AD.inference_utils import *

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

def get_scores(model_type, params, device):
    """
        Return anomaly scores, location, and gt label for patches for each orchard in a dictionary
        ARGS:
            model_type: string specifying the model to use (RD, RevisitingRD, UniAD, SCADN)
            params: dictionary containing parameters for the model
            device: device to run the model on
    """
    start_time = time.time()
    print("GETTING PATCH DATA...")
    score_dict = defaultdict(lambda: [])

    model = load_model(model_type, params, device)
    data_loader = get_loaders(model_type, params)
    if model_type == "RD" or model_type == "RevisitingRD":
        score_dict = get_scores_RD(model, data_loader, params, device)
    elif model_type == "UniAD":
        score_dict = get_scores_UniAD(model, data_loader, params, device)
    elif model_type == "SCADN":
        score_dict = get_scores_SCADN(model, data_loader, params, device)
    else:
        exit("[ERROR] UNKOWN MODEL")
    
    end_time = time.time()
    run_time = end_time - start_time
    print("TIME (s):", run_time)
    # write dict to file (only used for the demo) TODO
    #with open(f"ORCHARD_AD/{model_type}_score_dict.json", "w") as f:
    #    json.dump(score_dict, f)

    return score_dict


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

    hdbscan = HDBSCAN(min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"], cluster_selection_epsilon=params["epsilon"], alpha=params["alpha"])
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

        # relabel noise points to the closest non-noise point label
        noise_feature_indices = cluster_labels == -1
        non_noise_labels = cluster_labels[~noise_feature_indices]
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == -1:
                distances = np.linalg.norm(features_normalized[i] - features_normalized[~noise_feature_indices], axis=1) # calculate distance to all non-noise points
                cluster_labels[i] = non_noise_labels[np.argmin(distances)]    # assign the label of the closest non-noise point

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
        
        scatter = ax.scatter(features_normalized[:, 0], features_normalized[:, 1], features_normalized[:, 2], c=cluster_labels, cmap='rainbow')
        fig.colorbar(scatter)
        
        ax.set_title(f'3D Adaptive HDBSCAN Clustering for Orchard {orchard_id}', size=20)
        ax.set_xlabel('X Location', size=15)
        ax.set_ylabel('Y Location', size=15)
        ax.set_zlabel('Score', size=15)
        plt.show()
        
    return pr_dict, normal_cm, anomalous_cm


def tune_dbscan(params, trial):
    '''
        Perform orchard level inference using DBSCAN clustering
        ARGS:
            params: config dictionary
            score_dict: dictionary containing the scores for each patch for each orchard
    '''
    pr_dict = {}
    with open(f"data/{params['model_type']}_score_dict.json", "r") as f:
        score_dict = json.load(f)
    # initialize confusion matrices
    total_cm = np.zeros((2, 2))

    hdbscan = HDBSCAN(min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"], cluster_selection_epsilon=params["epsilon"], alpha=params["alpha"])
    for orchard_id, data in score_dict.items():
        scores = np.array([item[2] for item in data])           # get each patches score
        locations = np.array([item[0:2] for item in data])      # get each patches location
        gt_labels = np.array([item[3] for item in data])        # get each patches ground truth label (Anom or Normal) (used for evaluation only)
        #print(orchard_id, np.unique(gt_labels, return_counts=True))

        features = np.column_stack((locations, scores))
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        cluster_labels = hdbscan.fit_predict(features_normalized)     # fit the model
       
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
        # relabel noise points to the closest non-noise point label
        noise_feature_indices = cluster_labels == -1
        non_noise_labels = cluster_labels[~noise_feature_indices]
        for i in range(len(cluster_labels)):
            if cluster_labels[i] == -1:
                distances = np.linalg.norm(features_normalized[i] - features_normalized[~noise_feature_indices], axis=1) # calculate distance to all non-noise points
                cluster_labels[i] = non_noise_labels[np.argmin(distances)]    # assign the label of the closest non-noise point

        predictions = np.zeros(cluster_labels.shape)
        predictions[np.isin(cluster_labels, anom_clusters)] = -1
        predictions[np.isin(cluster_labels, norm_clusters)] = 1

        FP = np.sum((predictions == -1) & (gt_labels == 1))     # false positives
        FN = np.sum((predictions == 1) & (gt_labels == -1))     # false negatives
        TP = np.sum((predictions == -1) & (gt_labels == -1))    # true positives
        TN = np.sum((predictions == 1) & (gt_labels == 1))      # true negatives

        total_cm += np.array([[TN, FP], [FN, TP]])
    
    return get_F1(total_cm)

def objective(trial):
    parser = ArgumentParser(description="")
    parser.add_argument("--config", default="configs/inference.yaml", required=False)
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning with Optuna")
    args = parser.parse_args()

    config = args.config
    # open config
    with open(config, "r") as ymlfile:
        params = yaml.safe_load(ymlfile)

    # parameters to tune HDBSCAN
    params["min_cluster_size"] = trial.suggest_int("min_cluster_size", low=3, high=10)
    params["min_samples"] = trial.suggest_int("min_samples", low=3, high=10)
    params["epsilon"] = trial.suggest_float("epsilon", low=0.05, high=1.0)
    params["alpha"] = trial.suggest_float("alpha", low=0.1, high=1.5)
    params["v_thresh"] = trial.suggest_float("v_thresh", low=1.0, high=2.5)

    return tune_dbscan(params, trial)

def get_F1(cm):
    """
        Calculate the F1 score from a confusion matrix
    """
    tp, fp, fn, tn = cm.ravel()
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

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
    disp_norm = ConfusionMatrixDisplay(normal_cm, display_labels=["Normal", "Anomalous"])
    disp_anom.plot(values_format='', cmap='Blues')
    plt.title("Anomalous Orchards Confusion Matrix")
    plt.show()
    disp_norm.plot(values_format='', cmap='Blues')
    plt.title("Normal Orchards Confusion Matrix")
    plt.show()
    print("ANOMALOUS ORCHARD F1 SCORE:", get_F1(anomalous_cm))
    #print("NORMAL ORCHARD F1 SCORE:", get_F1(normal_cm))

if __name__ == "__main__":
    config_dir = "ORCHARD_AD/configs/"
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--orchard_config", "-oc", type=str, default="orchard_level_config.yaml")
    arg_parser.add_argument("--model_type", "-m", type=str, default="RevisitingRD", help="model type to use (RD, RevisitingRD, UniAD, SCADN)")
    arg_parser.add_argument("--model_config", "-mc", type=str, default="RD_config.yaml")
    arg_parser.add_argument("--test", action="store_true", help="load stored data from DL model instead of infering on each patch. Just here for making the demo faster")
    arg_parser.add_argument("--tune", action="store_true", help="for tuning HDBSCAN params with Optuna")
    args = arg_parser.parse_args()
    model_type = args.model_type
    with open(os.path.join(config_dir, args.orchard_config), "r") as f:
        orchard_params = yaml.safe_load(f)
    if model_type == "UniAD":
        model_params = load_config_UniAD(os.path.join(config_dir, args.model_config))
    else:
        with open(os.path.join(config_dir, args.model_config), "r") as f:
            model_params = yaml.safe_load(f)

    # tune the orchard level models WIP
    if args.tune:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1500)
        print(study.best_params)
        print(study.best_value)
        exit()
    else:
        pred_list = {-1: "Anomalous", 1: "Normal"}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load the json file and skip the DL model inference part WIP
        if args.test:
            with open(f"data/{model_type}_score_dict.json", "r") as f:
                score_dict = json.load(f)
        else:
            # start here
            score_dict = get_scores(model_type, model_params, device)

        pr_dict, normal_cm, anomalous_cm = infer_iso_forest(orchard_params, score_dict)
        print("===================== ISOLATION FOREST =====================")
        print_results(pr_dict, normal_cm, anomalous_cm)
        pr_dict, normal_cm, anomalous_cm = infer_dbscan(orchard_params, score_dict)
        print("========================== DBSCAN ==========================")
        print_results(pr_dict, normal_cm, anomalous_cm)
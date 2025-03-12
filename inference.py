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
    "2849": -1,
}

def get_scores(model_type, params, device='cuda'):
    """
        Return anomaly scores, location, and gt label for patches for each orchard in a dictionary
        ARGS:
            model_type: string specifying the model to use (RD, RevisitingRD, UniAD, SCADN)
            params: dictionary containing parameters for the model
            device: device to run the model on
    """
    # check if json file exists
    json_checkpoint = params["json_checkpoint"]
    if os.path.exists(f"ORCHARD_AD/checkpoints/{json_checkpoint}"):
        with open(f"ORCHARD_AD/checkpoints/{json_checkpoint}", "r") as f:
            return json.load(f)
    
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
    # write dict to file (only used for the demo)
    with open(f"ORCHARD_AD/checkpoints/{json_checkpoint}", "w") as f:
        json.dump(score_dict, f)

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
    anomalous_cm = defaultdict(lambda: np.zeros((2, 2)))
    
    # initialize isolation forest
    clf = IsolationForest(contamination=params["contamination"], random_state=42)
    print("NUMBER OF OUTLIERS TO NORMAL PATCHES DETECTED FOR EACH ORCHARD:")
    #clf.fit(np.concatenate([scores for scores in score_dict.values()]).reshape(-1, 1))    # fit the isolation forest
    for orchard_id, data in score_dict.items():
        scores = np.array([item[2] for item in data])           # get each patches score
        mean = np.array([item[3] for item in data])
        std = np.array([item[4] for item in data])
        complexity = np.array([item[5] for item in data])
        locations = np.array([item[0:2] for item in data])      # get each patches location
        gt_labels = np.array([item[-1] for item in data])        # get each patches ground truth label (Anom or Normal) (used for evaluation only)
        
        # combine scores and locations
        features = np.column_stack((scores, std, complexity))
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
            anomalous_cm[orchard_id] += np.array([[TN, FP], [FN, TP]])

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
    anomalous_cm = defaultdict(lambda: np.zeros((2, 2)))

    #score_dict = remap_coordinates(score_dict, sort_by_new=True)

    hdbscan = HDBSCAN(min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"], cluster_selection_epsilon=params["epsilon"], alpha=params["alpha"])
    for orchard_id, data in score_dict.items():
        scores = np.array([item[2] for item in data])           # get each patches score
        mean = np.array([item[3] for item in data])
        std = np.array([item[4] for item in data])
        complexity = np.array([item[5] for item in data])
        locations = np.array([item[0:2] for item in data])      # get each patches location
        gt_labels = np.array([item[-1] for item in data])        # get each patches ground truth label (Anom or Normal) (used for evaluation only)
        features = np.column_stack((locations, scores, std))
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        cluster_labels = hdbscan.fit_predict(features_normalized)     # fit the model
       
        # find the largest cluster (ignore noise / -1) and label it as normal
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
        if len(unique_labels) == 0:
            pr_dict[orchard_id] = 1
            continue
        else:
            normal_label = unique_labels[np.argmax(counts)]
            
            anom_clusters = []
            norm_clusters = [normal_label, -1]
            pr_dict[orchard_id] = 1
            noise_indices = cluster_labels == -1
            non_noise_features = features_normalized[~noise_indices]
            non_noise_labels = cluster_labels[~noise_indices]

            # normal mean used for thresholding later
            normal_mean = np.mean(features_normalized[cluster_labels == normal_label, 2])

            # first process noise points
            for i in range(len(cluster_labels)):
                # get nearest non noise cluster to this point
                distances = np.linalg.norm(features_normalized[i] - non_noise_features, axis=1)
                nearest_idx = np.argmin(distances)
                nearest_cluster = non_noise_labels[nearest_idx]
                # get mean and std of this cluster
                mean = np.mean(features_normalized[cluster_labels == nearest_cluster, 2])
                std = np.std(features_normalized[cluster_labels == nearest_cluster, 2])
                if cluster_labels[i] == -1:
                    # assign to a new cluster if score exceeds threshold
                    if features_normalized[i, 2] > mean + params["noise_thresh"] * std:
                        cluster_labels[i] = -2  # Special anomaly label
                    else:
                        # else assign to the nearest non noise cluster
                        cluster_labels[i] = nearest_cluster

            # classification decision, where you loop through each cluster and compare it's mean score to the largest cluster, if it's larger than threshold, classify orchard as anomalous
            for label in np.unique(cluster_labels):
                if label != normal_label and label != -1:
                    cluster_data = features_normalized[cluster_labels == label]
                    mean_score = np.mean(cluster_data[:, 2])
                    size = len(cluster_data)
                    if (mean_score > normal_mean + params["v_thresh"] and 
                        size >= params["min_cluster_size"]):
                        anom_clusters.append(label)
                        pr_dict[orchard_id] = -1
                    else:
                        #print(orchard_id, label, mean_score, size)
                        norm_clusters.append(label)

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
            anomalous_cm[orchard_id] += np.array([[TN, FP], [FN, TP]])
        
        # plot the clusters
        plot_orchard_clustering(features_normalized, cluster_labels, orchard_id, gt_labels)
        
    return pr_dict, normal_cm, anomalous_cm

def plot_orchard_clustering(features_normalized, cluster_labels, orchard_id, gt_labels):
    fig = plt.figure(figsize=(20, 10))
    
    # Clustering results subplot
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(features_normalized[:, 0], 
                          features_normalized[:, 1], 
                          features_normalized[:, 2], 
                          c=cluster_labels, 
                          cmap='brg')
    fig.colorbar(scatter1, ax=ax1)
    ax1.set_title(f'Clustering Results for Orchard {orchard_id}', size=20)
    ax1.set_xlabel('X Location', size=15)
    ax1.set_ylabel('Y Location', size=15)
    ax1.set_zlabel('Score', size=15)
    ax1.set_zlim(-3, 6)
    
    # ground truth subplot
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(features_normalized[:, 0], 
                          features_normalized[:, 1], 
                          features_normalized[:, 2], 
                          c=gt_labels, 
                          cmap='RdYlGn',  # Red for anomalous (-1), Green for normal (1)
                          vmin=-1, 
                          vmax=1)
    fig.colorbar(scatter2, ax=ax2)
    ax2.set_title(f'Ground Truth for Orchard {orchard_id}', size=20)
    ax2.set_xlabel('X Location', size=15)
    ax2.set_ylabel('Y Location', size=15)
    ax2.set_zlabel('Score', size=15)
    ax2.set_zlim(-3, 6)
    
    plt.tight_layout()
    #os.makedirs("ORCHARD_AD/plots", exist_ok=True)
    #plt.savefig(f"ORCHARD_AD/plots/{orchard_id}_clustering_comparison.png")
    plt.show()
    plt.close()

def tune_dbscan(params, score_dict, trial):
    '''
        Perform orchard level inference using DBSCAN clustering
        ARGS:
            params: config dictionary
            score_dict: dictionary containing the scores for each patch for each orchard
    '''
    pr_dict = {}
    # initialize confusion matrices
    normal_cm = np.zeros((2, 2))
    anomalous_cm = defaultdict(lambda: np.zeros((2, 2)))

    #score_dict = remap_coordinates(score_dict, sort_by_new=True)

    hdbscan = HDBSCAN(min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"], cluster_selection_epsilon=params["epsilon"], alpha=params["alpha"])
    for orchard_id, data in score_dict.items():
        scores = np.array([item[2] for item in data])           # get each patches score
        mean = np.array([item[3] for item in data])
        std = np.array([item[4] for item in data])
        complexity = np.array([item[5] for item in data])
        locations = np.array([item[0:2] for item in data])      # get each patches location
        gt_labels = np.array([item[-1] for item in data])        # get each patches ground truth label (Anom or Normal) (used for evaluation only)
        features = np.column_stack((locations, scores, std))
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        cluster_labels = hdbscan.fit_predict(features_normalized)     # fit the model
       
        # find the largest cluster (ignore noise / -1) and label it as normal
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)
        if len(unique_labels) == 0:
            pr_dict[orchard_id] = 1
            continue
        else:
            normal_label = unique_labels[np.argmax(counts)]
            
            anom_clusters = []
            norm_clusters = [normal_label, -1]
            pr_dict[orchard_id] = 1
            noise_indices = cluster_labels == -1
            non_noise_features = features_normalized[~noise_indices]
            non_noise_labels = cluster_labels[~noise_indices]

            # normal mean used for thresholding later
            normal_mean = np.mean(features_normalized[cluster_labels == normal_label, 2])

            # first process noise points
            for i in range(len(cluster_labels)):
                # get nearest non noise cluster to this point
                distances = np.linalg.norm(features_normalized[i] - non_noise_features, axis=1)
                nearest_idx = np.argmin(distances)
                nearest_cluster = non_noise_labels[nearest_idx]
                # get mean and std of this cluster
                mean = np.mean(features_normalized[cluster_labels == nearest_cluster, 2])
                std = np.std(features_normalized[cluster_labels == nearest_cluster, 2])
                if cluster_labels[i] == -1:
                    # assign to a new cluster if score exceeds threshold
                    if features_normalized[i, 2] > mean + params["noise_thresh"] * std:
                        cluster_labels[i] = -2  # Special anomaly label
                    else:
                        # else assign to the nearest non noise cluster
                        cluster_labels[i] = nearest_cluster

            # classification decision, where you loop through each cluster and compare it's mean score to the largest cluster, if it's larger than threshold, classify orchard as anomalous
            for label in np.unique(cluster_labels):
                if label != normal_label and label != -1:
                    cluster_data = features_normalized[cluster_labels == label]
                    mean_score = np.mean(cluster_data[:, 2])
                    size = len(cluster_data)
                    if (mean_score > normal_mean + params["v_thresh"] and 
                        size >= params["min_cluster_size"]):
                        anom_clusters.append(label)
                        pr_dict[orchard_id] = -1
                    else:
                        #print(orchard_id, label, mean_score, size)
                        norm_clusters.append(label)
    
        predictions = np.zeros(cluster_labels.shape)
        predictions[np.isin(cluster_labels, anom_clusters)] = -1
        predictions[np.isin(cluster_labels, norm_clusters)] = 1

        FP = np.sum((predictions == -1) & (gt_labels == 1))     # false positives
        FN = np.sum((predictions == 1) & (gt_labels == -1))     # false negatives
        TP = np.sum((predictions == -1) & (gt_labels == -1))    # true positives
        TN = np.sum((predictions == 1) & (gt_labels == 1))      # true negatives

        if gt_dict[orchard_id] == -1:
            anomalous_cm[orchard_id] += np.array([[TN, FP], [FN, TP]])
    
    orchard_acc = np.sum([1 for k, v in pr_dict.items() if v == gt_dict[k]]) / len(pr_dict)
    
    if orchard_acc != 1:
        return -1

    f1_dict = get_F1(anomalous_cm)
    return np.mean(list(f1_dict.values()))

def objective(trial, model_params, orchard_params, model_type):
    score_dict = get_scores(model_type, model_params)
    # parameters to tune HDBSCAN
    orchard_params["min_cluster_size"] = trial.suggest_int("min_cluster_size", low=3, high=10)
    orchard_params["min_samples"] = trial.suggest_int("min_samples", low=3, high=10)
    orchard_params["epsilon"] = trial.suggest_float("epsilon", low=0.05, high=1.0)
    orchard_params["alpha"] = trial.suggest_float("alpha", low=0.1, high=1.5)
    orchard_params["v_thresh"] = trial.suggest_float("v_thresh", low=1.0, high=2.5)
    orchard_params["noise_thresh"] = trial.suggest_float("noise_thresh", low=0.1, high=1.5)

    return tune_dbscan(orchard_params, score_dict, trial)

def get_F1(cm_dict):
    """
        Calculate the F1 score from a confusion matrix
    """
    f1 = defaultdict(float)
    for orchard_id, cm in cm_dict.items():
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            f1[orchard_id] = 0
        f1[orchard_id] = 2 * (precision * recall) / (precision + recall)

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
    
    anom_cm = np.sum(list(anomalous_cm.values()), axis=0)
    disp_anom = ConfusionMatrixDisplay(anom_cm, display_labels=["Normal", "Anomalous"])
    #print(anomalous_cm)
    #print(normal_cm)
    disp_norm = ConfusionMatrixDisplay(normal_cm, display_labels=["Normal", "Anomalous"])
    disp_anom.plot(values_format='', cmap='Blues')
    plt.title("Anomalous Orchards Confusion Matrix")
    plt.show()
    disp_norm.plot(values_format='', cmap='Blues')
    plt.title("Normal Orchards Confusion Matrix")
    plt.show()
    f1_dict = get_F1(anomalous_cm)
    for orchard_id, f1 in f1_dict.items():
        print(f"ORCHARD ID: {orchard_id}    F1 SCORE: {f1}")
    
    print("AVERAGE ANOMALOUS ORCHARD F1 SCORE:", np.mean(list(f1_dict.values())))

if __name__ == "__main__":
    os.makedirs("ORCHARD_AD/checkpoints", exist_ok=True)
    config_dir = "ORCHARD_AD/configs/"
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--orchard_config", "-oc", type=str, default="orchard_level_config.yaml")
    arg_parser.add_argument("--model_type", "-mt", type=str, default="RevisitingRD", help="model type to use (RD, RevisitingRD, UniAD, SCADN)")
    arg_parser.add_argument("--model_config", "-mc", type=str, default="RD_config.yaml")
    arg_parser.add_argument("--test", action="store_true", help="load stored data from DL model instead of infering on each patch. Just here for making the demo faster")
    arg_parser.add_argument("--tune", action="store_true", help="for tuning HDBSCAN params with Optuna")
    args = arg_parser.parse_args()
    model_type = args.model_type
    with open(os.path.join(config_dir, args.orchard_config), "r") as f:
        orchard_params = yaml.safe_load(f)
    
    model_config_path = os.path.join(config_dir, args.model_config)
    if model_type == "UniAD":
        model_params = load_config_UniAD(model_config_path)
    elif model_type == "SCADN":
        model_params = load_config_SCADN(model_config_path)
    elif model_type == "RD" or model_type == "RevisitingRD":
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f)

    # tune the orchard level models (assumes the json file exists)
    if args.tune:
        objective_w_params = lambda trial: objective(trial, model_params, orchard_params, model_type)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_w_params, n_trials=500)
        print(study.best_params)
        print(study.best_value)
        exit()
    else:
        pred_list = {-1: "Anomalous", 1: "Normal"}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # start here
        score_dict = get_scores(model_type, model_params, device)

        pr_dict, normal_cm, anomalous_cm = infer_iso_forest(orchard_params, score_dict)
        print("===================== ISOLATION FOREST =====================")
        print_results(pr_dict, normal_cm, anomalous_cm)
        pr_dict, normal_cm, anomalous_cm = infer_dbscan(orchard_params, score_dict)
        print("========================== DBSCAN ==========================")
        print_results(pr_dict, normal_cm, anomalous_cm)
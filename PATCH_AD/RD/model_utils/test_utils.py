import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
import pickle
from model_utils.plots import plot_sample, plot_histogram
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

from collections import defaultdict

def cal_anomaly_map(fs_list, ft_list, out_size=256, amap_mode='mul', weights=[1.0, 1.0, 1.0]):
    """
        calculate anomaly map by comparing feature maps from encoder and decoder. 
        amap_mode is either 'mul' or 'add' indicating whether to multiply or add the anomaly maps from each layer
    """
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        # similar to loss_function in main.py, calculate cosine similarity between corresponding feature maps but then interpolate to the original image size
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = weights[i] * (1 - F.cosine_similarity(fs, ft))
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def cal_anomaly_score(inputs, outputs, score_weight=0, out_size=256, feature_weights=[1.0, 1.0, 1.0]):
    anomaly_map, _ = cal_anomaly_map(inputs, outputs, out_size, amap_mode='a', weights=feature_weights)
    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
    anomaly_score = np.max(anomaly_map) + score_weight * np.average(anomaly_map)

    return anomaly_score

def get_orchard_stats(orchard_data):
    auroc_case1 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case_1"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case_1"])
    auroc_case2 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case_2"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case_2"])
    auroc_case3 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case_3"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case_3"])
    
    stat_dict = defaultdict(lambda: [0, 0])  # mean and std dev for each case and normal
    
    stat_dict["normal"][0] = round(sum(orchard_data["pr_normal"]) / len(orchard_data["pr_normal"]), 5)
    stat_dict["normal"][1] = round(np.std(orchard_data["pr_normal"]), 5)
    stat_dict["case_2"][0] = round(sum(orchard_data["pr_case_2"]) / len(orchard_data["pr_case_2"]), 5) if not len(orchard_data["pr_case_2"]) == 0 else None
    stat_dict["case_2"][1] = round(np.std(orchard_data["pr_case_2"]), 5) if not len(orchard_data["pr_case_2"]) == 0 else None
    stat_dict["case_1"][0] = round(sum(orchard_data["pr_case_1"]) / len(orchard_data["pr_case_1"]), 5) if not len(orchard_data["pr_case_1"]) == 0 else None
    stat_dict["case_1"][1] = round(np.std(orchard_data["pr_case_1"]), 5) if not len(orchard_data["pr_case_1"]) == 0 else None
    stat_dict["case_3"][0] = round(sum(orchard_data["pr_case_3"]) / len(orchard_data["pr_case_3"]), 5) if not len(orchard_data["pr_case_3"]) == 0 else None
    stat_dict["case_3"][1] = round(np.std(orchard_data["pr_case_3"]), 5) if not len(orchard_data["pr_case_3"]) == 0 else None

    return stat_dict, auroc_case1, auroc_case2, auroc_case3

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def evaluate_RD(model, data_loader, device, log_path = None, score_weight = 1.0, feature_weights=[1.0, 1.0, 1.0]):
    """
    Evaluate the model for multiple anomaly types
    Returns: average_auroc, orchard_auroc_dict
    """
    # average auroc for each orchard
    orchard_anomaly_score_dict = defaultdict(lambda: {"pr_case_1": [], "pr_case_2": [], "pr_case_3": [], "pr_normal": []})     # dict of orchard id anomaly scores for each case and normal patch. Default value supplied if key not found
    orchard_auroc_dict = {}

    model.eval()
    with torch.no_grad():
        for input in data_loader:
            img = input["image"].to(device)
            #img = img.squeeze(-1)
            #label = input["label"].item()   # 0 for normal, 1 for artefact
            cls_name, case_id = input["clsname"][0], input["case"][0]   # cls_name either "all" or orchard id
            #print(f"[INFO] CLASS NAME: {cls_name}, CASE NUM: {case_id}")

            inputs, outputs = model(img)
            anomaly_score = cal_anomaly_score(inputs, outputs, score_weight, img.shape[-1], feature_weights)
            orchard_anomaly_score_dict[cls_name][f"pr_{case_id}"].append(anomaly_score)
   
    # calculate AUROC for each case for each orchard
    if log_path :
        with open(log_path, "a") as file:
            file.write("\n=== EVALUATION ===")
    
    for orchard_id, orchard_data in orchard_anomaly_score_dict.items():
        # get stats for each case
        orchard_score_stats, auroc_case  = get_orchard_stats(orchard_data)

        #auroc_total = sum([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None]) / len([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None])
        #orchard_auroc_dict[orchard_id] = auroc_total * 100
        
        if log_path:
            with open(log_path, "a") as file:
                    file.write(f"\n-- {orchard_id}, CASE 1 AUROC: {auroc_case1}, CASE 2 AUROC: {auroc_case2}, CASE 3 AUROC: {auroc_case3} OVERALL: {auroc_total}")
                    file.write(f"\n++ NORMAL MEAN: {orchard_score_stats['normal'][0]} STD DEV: {orchard_score_stats['normal'][1]}"+
                           f"\n++ CASE 1 MEAN: {orchard_score_stats['case_1'][0]} STD DEV: {orchard_score_stats['case_1'][1]}" +
                           f"\n++ CASE 2 MEAN: {orchard_score_stats['case_2'][0]} STD DEV: {orchard_score_stats['case_2'][1]}" +
                           f"\n++ CASE 3 MEAN: {orchard_score_stats['case_3'][0]} STD DEV: {orchard_score_stats['case_3'][1]}")

    # calculate AUROC overall
    average_auroc = sum([x for x in orchard_auroc_dict.values()]) / len(orchard_auroc_dict)
    orchard_auroc_dict["total"] = average_auroc * 100
    if log_path:
        with open(log_path, "a") as file:
            file.write(f"\n@@ Average AUROC: {average_auroc}")
            file.write(f"\n==================\n")
    
    return average_auroc, orchard_auroc_dict

def test_RD(model, data_loader, device, model_path, score_weight = 1.0, feature_weights = [1.0, 1.0, 1.0], n_plot_per_class=0):
    """
        Load the best model state after training, evaluate it at the patch level and then plot per orchard histograms and precision-recall curves
    """
    orchard_anomaly_score_dict = defaultdict(lambda: {"pr_case_1": [], "pr_case_2": [], "pr_case_3": [], "pr_normal": []})     # dict of orchard id anomaly scores for each case and normal patch. Default value supplied if key not found
    plot_count = {"1676": {"case_2": 0, "normal": 0}, "1996": {"case_1": 0, "case_2": 0, "normal": 0}, "2057": {"case_1": 0, "case_2": 0, "normal": 0}, "all": {"case_1": 0, "case_2": 0, "normal": 0}}  # dict of orchard id and label count for plotting
    orchard_patch_results = {"1676": {"score": [], "img": [], "label": []}, "1996": {"score": [], "img": [], "label": []}, "2057": {"score": [], "img": [], "label": []}}
    orchard_auroc_dict = {}
    # load the best model after training
    model.load_model(model_path)
    model.eval()

    with torch.no_grad():
        for input in data_loader:
            img = input["image"].to(device)

            cls_name, case_id = input["clsname"][0].split('_')[0], input["case"][0]   # cls_name either "all" or orchard id
            #print(f"[INFO] CLASS NAME: {cls_name}, CASE NUM: {case_id}")

            inputs, outputs = model(img)
            anomaly_score = cal_anomaly_score(inputs, outputs, score_weight, img.shape[-1], feature_weights)
            orchard_anomaly_score_dict[cls_name][f"pr_{case_id}"].append(anomaly_score)

            if not cls_name == "all":
                if plot_count[cls_name][case_id] < n_plot_per_class:
                    plot_count[cls_name][case_id] += 1
                    orchard_patch_results[cls_name]["score"].append(anomaly_score)
                    orchard_patch_results[cls_name]["img"].append(img)
                    orchard_patch_results[cls_name]["label"].append(case_id)
    
    print("[INFO] FINAL RESULTS:")
    for orchard_id, orchard_data in orchard_anomaly_score_dict.items():
        # calc average scores for each case
        orchard_score_stats, auroc_case1, auroc_case2, auroc_case3  = get_orchard_stats(orchard_data)
        auroc_total = sum([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None]) / len([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None])
        orchard_auroc_dict[orchard_id] = auroc_total * 100
        print(f"- ID: {orchard_id}, CASE 1 AUROC: {auroc_case1}, CASE 2 AUROC: {auroc_case2}, CASE 3 AUROC: {auroc_case3} OVERALL: {auroc_total}")
        print(f"++ NORMAL MEAN: {orchard_score_stats['normal'][0]} STD DEV: {orchard_score_stats['normal'][1]}"+
                           f"\n++ CASE 1 MEAN: {orchard_score_stats['case_1'][0]} STD DEV: {orchard_score_stats['case_1'][1]}" +
                           f"\n++ CASE 2 MEAN: {orchard_score_stats['case_2'][0]} STD DEV: {orchard_score_stats['case_2'][1]}" +
                           f"\n++ CASE 3 MEAN: {orchard_score_stats['case_3'][0]} STD DEV: {orchard_score_stats['case_3'][1]}")
        
        plot_histogram(orchard_data["pr_case_1"], orchard_data["pr_case_2"], orchard_data["pr_case_3"], orchard_data["pr_normal"], orchard_id)

        fig, ax = plt.subplots()
        if auroc_case1:
            PrecisionRecallDisplay.from_predictions([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case_1"]))],
                                                        orchard_data["pr_normal"] + orchard_data["pr_case_1"], name="CASE 1 PRECISION RECALL", ax=ax)
        if auroc_case2:
            PrecisionRecallDisplay.from_predictions([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case_2"]))],
                                                        orchard_data["pr_normal"] + orchard_data["pr_case_2"], name="CASE 2 PRECISION RECALL", ax=ax)
        if auroc_case3:
            PrecisionRecallDisplay.from_predictions([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case_3"]))],
                                                        orchard_data["pr_normal"] + orchard_data["pr_case_3"], name="CASE 3 PRECISION RECALL", ax=ax)
        plt.show()
        if not orchard_id == "all":        
            for i in range(len(orchard_patch_results[orchard_id]["img"])):
                plot_sample(orchard_patch_results[orchard_id]["img"][i], orchard_patch_results[orchard_id]["label"][i], orchard_patch_results[orchard_id]["score"][i], orchard_id)

    print(f"[INFO] FINAL AVERAGE AUROC: {sum([x for x in orchard_auroc_dict.values()]) / len(orchard_auroc_dict)}")

def calculate_auroc(gt_list, pr_list):
    if len(set(gt_list)) == 2:
        auroc = round(roc_auc_score(gt_list, pr_list), 3)
        return auroc
    else:
        return None
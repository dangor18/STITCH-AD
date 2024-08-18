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

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul', weights=[1.0, 1.0, 1.0]):
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

def evaluation(encoder, bn, decoder, data_loader, device, log_path = None, score_weight = 1.0, feature_weights=[1.0, 1.0, 1.0]):
    """
    Evaluate the model for multiple anomaly types
    """
    bn.eval()
    decoder.eval()

    # average auroc for each orchard
    average_auroc = 0
    orchard_count = 0
    max_anomaly_score = 0
    orchard_auroc_results = {}
    orchard_case_count = {}
    orchard_anomaly_scores = {}
    orchard_auroc_dict = {}

    with torch.no_grad():
        for input in data_loader:
            img = input["image"].to(device)
            label = input["label"].item()
            orchard_id = input["clsname"][0]

            if orchard_auroc_results.get(orchard_id) is None:
                orchard_auroc_results[orchard_id] = {"pr_case1": [], "pr_case2": [], "pr_case3": [], "pr_normal": []}
                orchard_case_count[orchard_id] = {"normal": 0, "case_1": 0, "case_2": 0, "case_3": 0}
                orchard_anomaly_scores[orchard_id] = {"normal": [0,0], "case_1": [0,0], "case_2": [0,0], "case_3": [0,0]}    # first for mean second for std dev of anomaly scores

            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a', weights=feature_weights)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
           
            anomaly_score = np.max(anomaly_map) + score_weight * np.average(anomaly_map)
            #max_anomaly_score = max(max_anomaly_score, anomaly_score)
           
            if label == 1:  # case_1 anomaly
                orchard_auroc_results[orchard_id]["pr_case1"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_1"] += 1
            elif label == 2:  # case_2 anomaly
                orchard_auroc_results[orchard_id]["pr_case2"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_2"] += 1
            elif label == 3:  # case_3 anomaly
                orchard_auroc_results[orchard_id]["pr_case3"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_3"] += 1
            elif label == 0:  # normal
                orchard_auroc_results[orchard_id]["pr_normal"].append(anomaly_score)
                orchard_case_count[orchard_id]["normal"] += 1
   
    # calculate AUROC for each case for each orchard
    if log_path :
        with open(log_path, "a") as file:
            file.write("\n=== EVALUATION ===")
    
    for orchard_id, orchard_data in orchard_auroc_results.items():
        #print(orchard_id)
        # normalize scores with max anomaly score
        #for key in ["pr_case1", "pr_case2", "pr_normal"]:
            #orchard_data[key] = [score / max_anomaly_score for score in orchard_data[key]]

        # calc average scores for each case
        orchard_anomaly_scores[orchard_id]["normal"][0] = round(sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"], 5)
        orchard_anomaly_scores[orchard_id]["normal"][1] = round(np.std(orchard_data["pr_normal"]), 5)
        # "flip" anomaly scores for case 1 when they are too low (by first getting the difference between the average normal score and then adding the diff. to this) 
        '''
        if (orchard_case_count[orchard_id]["case_1"] != 0 and 
        sum(orchard_data["pr_case1"]) / orchard_case_count[orchard_id]["case_1"] 
        < sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"]):
            orchard_data["pr_case1"] = [orchard_anomaly_scores[orchard_id]["normal"][0] - x + orchard_anomaly_scores[orchard_id]["normal"][0] for x in orchard_data["pr_case1"]]
        if (orchard_case_count[orchard_id]["case_3"] != 0 and 
        sum(orchard_data["pr_case3"]) / orchard_case_count[orchard_id]["case_3"] 
        < sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"]):
            orchard_data["pr_case3"] = [orchard_anomaly_scores[orchard_id]["normal"][0] - x + orchard_anomaly_scores[orchard_id]["normal"][0] for x in orchard_data["pr_case3"]]
        '''
        orchard_anomaly_scores[orchard_id]["case_2"][0] = round(sum(orchard_data["pr_case2"]) / orchard_case_count[orchard_id]["case_2"], 5) if not len(orchard_data["pr_case2"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_2"][1] = round(np.std(orchard_data["pr_case2"]), 5) if not len(orchard_data["pr_case2"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_1"][0] = round(sum(orchard_data["pr_case1"]) / orchard_case_count[orchard_id]["case_1"], 5) if not len(orchard_data["pr_case1"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_1"][1] = round(np.std(orchard_data["pr_case1"]), 5) if not len(orchard_data["pr_case1"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_3"][0] = round(sum(orchard_data["pr_case3"]) / orchard_case_count[orchard_id]["case_3"], 5) if not len(orchard_data["pr_case3"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_3"][1] = round(np.std(orchard_data["pr_case3"]), 5) if not len(orchard_data["pr_case3"]) == 0 else None
        auroc_case1 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case1"])
        auroc_case2 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case2"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case2"])
        auroc_case3 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case3"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case3"])
        #auroc_total = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"] + orchard_data["pr_case2"]))], 
                                      #orchard_data["pr_normal"] + orchard_data["pr_case1"] + orchard_data["pr_case2"])
        auroc_total = sum([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None]) / len([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None])
        orchard_auroc_dict[orchard_id] = auroc_total * 100
        #if auroc_case1:
            #auroc_total = (auroc_case1 + auroc_case2) / 2
        #else:
            #auroc_total = auroc_case2
        average_auroc += auroc_total
        orchard_count += 1
        if log_path:
            with open(log_path, "a") as file:
                file.write(f"\n-- ID: {orchard_id}, CASE 1 AUROC: {auroc_case1}, CASE 2 AUROC: {auroc_case2}, CASE 3 AUROC: {auroc_case3} OVERALL: {auroc_total}")
                file.write(f"\n++ NORMAL MEAN: {orchard_anomaly_scores[orchard_id]['normal'][0]} STD DEV: {orchard_anomaly_scores[orchard_id]['normal'][1]}"+
                           f"\n++ CASE 1 MEAN: {orchard_anomaly_scores[orchard_id]['case_1'][0]} STD DEV: {orchard_anomaly_scores[orchard_id]['case_1'][1]}" +
                           f"\n++ CASE 2 MEAN: {orchard_anomaly_scores[orchard_id]['case_2'][0]} STD DEV: {orchard_anomaly_scores[orchard_id]['case_2'][1]}" +
                           f"\n++ CASE 3 MEAN: {orchard_anomaly_scores[orchard_id]['case_3'][0]} STD DEV: {orchard_anomaly_scores[orchard_id]['case_3'][1]}")

    # calculate AUROC overall
    average_auroc = round(average_auroc / orchard_count, 5)
    orchard_auroc_dict["total"] = average_auroc * 100
    if log_path:
        with open(log_path, "a") as file:
            file.write(f"\n@@ Average AUROC: {average_auroc}")
            file.write(f"\n==================\n")
    
    return average_auroc, orchard_auroc_dict

def test(encoder, bn, decoder, data_loader, device, model_path, score_weight = 1.0, feature_weights = [1.0, 1.0, 1.0], n_plot_per_class=0):
    """
        Load the best model state after training, evaluate it at the patch level and then plot per orchard histograms and precision-recall curves
    """
    # load the best model after training
    ckp = torch.load(model_path, weights_only=True)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    
    bn.eval()
    decoder.eval()
   
    # average auroc for each orchard
    orchard_patch_results = {}
    orchard_case_count = {}
    orchard_anomaly_scores = {}
    orchard_auroc_results = {}
    plot_count = {}

    with torch.no_grad():
        for input in data_loader:
            img = input["image"].to(device)
            label = input["label"].item()
            orchard_id = input["clsname"][0]

            if orchard_patch_results.get(orchard_id) is None:
                orchard_auroc_results[orchard_id] = {"pr_case1": [], "pr_case2": [], "pr_case3": [], "pr_normal": []}
                orchard_patch_results[orchard_id] = {"img": [], "label": [], "score": []}
                orchard_case_count[orchard_id] = {"normal": 0, "case_1": 0, "case_2": 0, "case_3": 0}
                orchard_anomaly_scores[orchard_id] = {"normal": [0,0], "case_1": [0,0], "case_2": [0,0], "case_3": [0,0]}    # first for mean second for std dev of anomaly scores
                plot_count[orchard_id] = {0: 0, 1: 0, 2: 0, 3: 0}

            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a', weights=feature_weights)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
           
            anomaly_score = np.max(anomaly_map) + score_weight * np.average(anomaly_map)
            #max_anomaly_score = max(max_anomaly_score, anomaly_score)
            if plot_count[orchard_id][label] < n_plot_per_class:
                plot_count[orchard_id][label] += 1
                orchard_patch_results[orchard_id]["score"].append(anomaly_score)
                orchard_patch_results[orchard_id]["img"].append(img)
                orchard_patch_results[orchard_id]["label"].append(label)
            
            if label == 1:  # case_1 anomaly
                orchard_auroc_results[orchard_id]["pr_case1"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_1"] += 1
            elif label == 2:  # case_2 anomaly
                orchard_auroc_results[orchard_id]["pr_case2"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_2"] += 1
            elif label == 3:  # case_3 anomaly
                orchard_auroc_results[orchard_id]["pr_case3"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_3"] += 1
            elif label == 0:  # normal
                orchard_auroc_results[orchard_id]["pr_normal"].append(anomaly_score)
                orchard_case_count[orchard_id]["normal"] += 1
    
    for orchard_id, orchard_data in orchard_auroc_results.items():
        # normalize scores with max anomaly score
        #for key in ["pr_case1", "pr_case2", "pr_normal"]:
            #orchard_data[key] = [score / max_anomaly_score for score in orchard_data[key]]

        # calc average scores for each case
        orchard_anomaly_scores[orchard_id]["normal"][0] = round(sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"], 5)
        orchard_anomaly_scores[orchard_id]["normal"][1] = round(np.std(orchard_data["pr_normal"]), 5)
        # "flip" anomaly scores for case 1 when they are too low (by first getting the difference between the average normal score and then adding the diff. to this) 
        '''
        if (orchard_case_count[orchard_id]["case_1"] != 0 and 
        sum(orchard_data["pr_case1"]) / orchard_case_count[orchard_id]["case_1"] 
        < sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"]):
            orchard_data["pr_case1"] = [orchard_anomaly_scores[orchard_id]["normal"][0] - x + orchard_anomaly_scores[orchard_id]["normal"][0] for x in orchard_data["pr_case1"]]
            for i in range(len(orchard_patch_results[orchard_id]["img"])):
                if orchard_patch_results[orchard_id]["label"][i] == 1:
                    orchard_patch_results[orchard_id]["score"][i] = orchard_anomaly_scores[orchard_id]["normal"][0] - orchard_patch_results[orchard_id]["score"][i] + orchard_anomaly_scores[orchard_id]["normal"][0]
        '''
        '''
        if (orchard_case_count[orchard_id]["case_2"] != 0 and 
        sum(orchard_data["pr_case2"]) / orchard_case_count[orchard_id]["case_2"] 
        < sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"]):
            orchard_data["pr_case2"] = [orchard_anomaly_scores[orchard_id]["normal"][0] - x + orchard_anomaly_scores[orchard_id]["normal"][0] for x in orchard_data["pr_case2"]]
            for i in range(len(orchard_patch_results[orchard_id]["img"])):
                if orchard_patch_results[orchard_id]["label"][i] == 2:
                    orchard_patch_results[orchard_id]["score"][i] = orchard_anomaly_scores[orchard_id]["normal"][0] - orchard_patch_results[orchard_id]["score"][i] + orchard_anomaly_scores[orchard_id]["normal"][0]
        '''
        auroc_case1 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case1"])
        auroc_case2 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case2"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case2"])
        auroc_case3 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case3"]))],
                                        orchard_data["pr_normal"] + orchard_data["pr_case3"])
        #auroc_total = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"] + orchard_data["pr_case2"]))], 
                                      #orchard_data["pr_normal"] + orchard_data["pr_case1"] + orchard_data["pr_case2"])
        auroc_total = sum([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None]) / len([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None])
        print("[INFO] FINAL RESULTS:")
        print(f"- ID: {orchard_id}, CASE 1 AUROC: {auroc_case1}, CASE 2 AUROC: {auroc_case2}, CASE 3 AUROC: {auroc_case3} OVERALL: {auroc_total}")
        plot_histogram(orchard_data["pr_case1"], orchard_data["pr_case2"], orchard_data["pr_case3"], orchard_data["pr_normal"], orchard_id)

        fig, ax = plt.subplots()
        if auroc_case1:
            PrecisionRecallDisplay.from_predictions([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"]))],
                                                        orchard_data["pr_normal"] + orchard_data["pr_case1"], name="CASE 1 PRECISION RECALL", ax=ax)
        if auroc_case2:
            PrecisionRecallDisplay.from_predictions([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case2"]))],
                                                        orchard_data["pr_normal"] + orchard_data["pr_case2"], name="CASE 2 PRECISION RECALL", ax=ax)
        if auroc_case3:
            PrecisionRecallDisplay.from_predictions([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case3"]))],
                                                        orchard_data["pr_normal"] + orchard_data["pr_case3"], name="CASE 3 PRECISION RECALL", ax=ax)
        plt.show()
        if n_plot_per_class > 0:        
            for i in range(len(orchard_patch_results[orchard_id]["img"])):
                plot_sample(orchard_patch_results[orchard_id]["img"][i], orchard_patch_results[orchard_id]["label"][i], orchard_patch_results[orchard_id]["score"][i], orchard_id)

def evaluation_multi_proj(encoder, proj, bn, decoder, data_loader, device, log_path = None, score_weight = 1.0, feature_weights=[1.0, 1.0, 1.0]):
    """
        Evaluate the model for multiple anomaly types
    """
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()

    average_auroc = 0
    orchard_count = 0
    max_anomaly_score = 0
    orchard_auroc_results = {}
    orchard_case_count = {}
    orchard_anomaly_scores = {}
    orchard_auroc_dict = {}

    with torch.no_grad():
        for input in data_loader:
            img = input["image"].to(device)
            label = input["label"].item()
            orchard_id = input["clsname"][0]

            if orchard_auroc_results.get(orchard_id) is None:
                orchard_auroc_results[orchard_id] = {"pr_case1": [], "pr_case2": [], "pr_case3": [], "pr_normal": []}
                orchard_case_count[orchard_id] = {"normal": 0, "case_1": 0, "case_2": 0, "case_3": 0}
                orchard_anomaly_scores[orchard_id] = {"normal": [0,0], "case_1": [0,0], "case_2": [0,0], "case_3": [0,0]}    # first for mean second for std dev of anomaly scores

            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a', weights=feature_weights)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
           
            anomaly_score = np.max(anomaly_map) + score_weight *  np.average(anomaly_map)
            #max_anomaly_score = max(max_anomaly_score, anomaly_score)
           
            if label == 1:  # case_1 anomaly
                orchard_auroc_results[orchard_id]["pr_case1"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_1"] += 1
            elif label == 2:  # case_2 anomaly
                orchard_auroc_results[orchard_id]["pr_case2"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_2"] += 1
            elif label == 3:  # case_3 anomaly
                orchard_auroc_results[orchard_id]["pr_case3"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_3"] += 1
            elif label == 0:  # normal
                orchard_auroc_results[orchard_id]["pr_normal"].append(anomaly_score)
                orchard_case_count[orchard_id]["normal"] += 1
   
    # calculate AUROC for each case for each orchard
    if log_path :
        with open(log_path, "a") as file:
            file.write("\n=== EVALUATION ===")
    
    for orchard_id, orchard_data in orchard_auroc_results.items():
        #print(orchard_id)
        # normalize scores with max anomaly score
        #for key in ["pr_case1", "pr_case2", "pr_normal"]:
            #orchard_data[key] = [score / max_anomaly_score for score in orchard_data[key]]

        # calc average scores for each case
        orchard_anomaly_scores[orchard_id]["normal"][0] = round(sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"], 5)
        orchard_anomaly_scores[orchard_id]["normal"][1] = round(np.std(orchard_data["pr_normal"]), 5)
        # "flip" anomaly scores for case 1 when they are too low (by first getting the difference between the average normal score and then adding the diff. to this) 
        '''
        if (orchard_case_count[orchard_id]["case_1"] != 0 and 
        sum(orchard_data["pr_case1"]) / orchard_case_count[orchard_id]["case_1"] 
        < sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"]):
            orchard_data["pr_case1"] = [orchard_anomaly_scores[orchard_id]["normal"][0] - x + orchard_anomaly_scores[orchard_id]["normal"][0] for x in orchard_data["pr_case1"]]
        if (orchard_case_count[orchard_id]["case_3"] != 0 and 
        sum(orchard_data["pr_case3"]) / orchard_case_count[orchard_id]["case_3"] 
        < sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"]):
            orchard_data["pr_case3"] = [orchard_anomaly_scores[orchard_id]["normal"][0] - x + orchard_anomaly_scores[orchard_id]["normal"][0] for x in orchard_data["pr_case3"]]
        if (orchard_case_count[orchard_id]["case_2"] != 0 and 
        sum(orchard_data["pr_case2"]) / orchard_case_count[orchard_id]["case_2"] 
        < sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"]):
            orchard_data["pr_case2"] = [orchard_anomaly_scores[orchard_id]["normal"][0] - x + orchard_anomaly_scores[orchard_id]["normal"][0] for x in orchard_data["pr_case2"]]
        '''
        orchard_anomaly_scores[orchard_id]["case_2"][0] = round(sum(orchard_data["pr_case2"]) / orchard_case_count[orchard_id]["case_2"], 5) if not len(orchard_data["pr_case2"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_2"][1] = round(np.std(orchard_data["pr_case2"]), 5) if not len(orchard_data["pr_case2"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_1"][0] = round(sum(orchard_data["pr_case1"]) / orchard_case_count[orchard_id]["case_1"], 5) if not len(orchard_data["pr_case1"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_1"][1] = round(np.std(orchard_data["pr_case1"]), 5) if not len(orchard_data["pr_case1"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_3"][0] = round(sum(orchard_data["pr_case3"]) / orchard_case_count[orchard_id]["case_3"], 5) if not len(orchard_data["pr_case3"]) == 0 else None
        orchard_anomaly_scores[orchard_id]["case_3"][1] = round(np.std(orchard_data["pr_case3"]), 5) if not len(orchard_data["pr_case3"]) == 0 else None
        auroc_case1 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case1"])
        auroc_case2 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case2"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case2"])
        auroc_case3 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case3"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case3"])
        #auroc_total = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"] + orchard_data["pr_case2"]))], 
                                      #orchard_data["pr_normal"] + orchard_data["pr_case1"] + orchard_data["pr_case2"])
        auroc_total = sum([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None]) / len([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None])
        orchard_auroc_dict[orchard_id] = auroc_total * 100
        #if auroc_case1:
            #auroc_total = (auroc_case1 + auroc_case2) / 2
        #else:
            #auroc_total = auroc_case2
        average_auroc += auroc_total
        orchard_count += 1
        if log_path:
            with open(log_path, "a") as file:
                file.write(f"\n-- ID: {orchard_id}, CASE 1 AUROC: {auroc_case1}, CASE 2 AUROC: {auroc_case2}, CASE 3 AUROC: {auroc_case3} OVERALL: {auroc_total}")
                file.write(f"\n++ NORMAL MEAN: {orchard_anomaly_scores[orchard_id]['normal'][0]} STD DEV: {orchard_anomaly_scores[orchard_id]['normal'][1]}"+
                           f"\n++ CASE 1 MEAN: {orchard_anomaly_scores[orchard_id]['case_1'][0]} STD DEV: {orchard_anomaly_scores[orchard_id]['case_1'][1]}" +
                           f"\n++ CASE 2 MEAN: {orchard_anomaly_scores[orchard_id]['case_2'][0]} STD DEV: {orchard_anomaly_scores[orchard_id]['case_2'][1]}" +
                           f"\n++ CASE 3 MEAN: {orchard_anomaly_scores[orchard_id]['case_3'][0]} STD DEV: {orchard_anomaly_scores[orchard_id]['case_3'][1]}")

    # calculate AUROC overall
    average_auroc = round(average_auroc / orchard_count, 5)
    orchard_auroc_dict["total"] = average_auroc * 100
    if log_path:
        with open(log_path, "a") as file:
            file.write(f"\n@@ Average AUROC: {average_auroc}")
            file.write(f"\n==================\n")
    
    return average_auroc, orchard_auroc_dict

def test_multi_proj(encoder, proj, bn, decoder, data_loader, device, model_path, score_weight = 1.0, feature_weights = [1.0, 1.0, 1.0], n_plot_per_class=0):
    """
        Load the best model state after training, evaluate it at the patch level and then plot per orchard histograms and precision-recall curves
    """
    # load the best model after training
    ckp = torch.load(model_path, weights_only=True)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    proj.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    
    bn.eval()
    proj.eval()
    decoder.eval()
   
    # average auroc for each orchard
    orchard_patch_results = {}
    orchard_case_count = {}
    orchard_anomaly_scores = {}
    orchard_auroc_results = {}
    plot_count = {}

    with torch.no_grad():
        for input in data_loader:
            img = input["image"].to(device)
            label = input["label"].item()
            orchard_id = input["clsname"][0]

            if orchard_patch_results.get(orchard_id) is None:
                orchard_auroc_results[orchard_id] = {"pr_case1": [], "pr_case2": [], "pr_case3": [], "pr_normal": []}
                orchard_patch_results[orchard_id] = {"img": [], "label": [], "score": []}
                orchard_case_count[orchard_id] = {"normal": 0, "case_1": 0, "case_2": 0, "case_3": 0}
                orchard_anomaly_scores[orchard_id] = {"normal": [0,0], "case_1": [0,0], "case_2": [0,0], "case_3": [0,0]}    # first for mean second for std dev of anomaly scores
                plot_count[orchard_id] = {0: 0, 1: 0, 2: 0, 3: 0}

            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a', weights=feature_weights)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
           
            anomaly_score = np.max(anomaly_map) + score_weight * np.average(anomaly_map)
            #max_anomaly_score = max(max_anomaly_score, anomaly_score)
            if plot_count[orchard_id][label] < n_plot_per_class:
                plot_count[orchard_id][label] += 1
                orchard_patch_results[orchard_id]["score"].append(anomaly_score)
                orchard_patch_results[orchard_id]["img"].append(img)
                orchard_patch_results[orchard_id]["label"].append(label)
            
            if label == 1:  # case_1 anomaly
                orchard_auroc_results[orchard_id]["pr_case1"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_1"] += 1
            elif label == 2:  # case_2 anomaly
                orchard_auroc_results[orchard_id]["pr_case2"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_2"] += 1
            elif label == 3:  # case_3 anomaly
                orchard_auroc_results[orchard_id]["pr_case3"].append(anomaly_score)
                orchard_case_count[orchard_id]["case_3"] += 1
            elif label == 0:  # normal
                orchard_auroc_results[orchard_id]["pr_normal"].append(anomaly_score)
                orchard_case_count[orchard_id]["normal"] += 1
    
    for orchard_id, orchard_data in orchard_auroc_results.items():
        # calc average scores for each case
        orchard_anomaly_scores[orchard_id]["normal"][0] = round(sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"], 5)
        orchard_anomaly_scores[orchard_id]["normal"][1] = round(np.std(orchard_data["pr_normal"]), 5)
        # "flip" anomaly scores for case 1 when they are too low (by first getting the difference between the average normal score and then adding the diff. to this) 
        '''
        if (orchard_case_count[orchard_id]["case_1"] != 0 and 
        sum(orchard_data["pr_case1"]) / orchard_case_count[orchard_id]["case_1"] 
        < sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"]):
            orchard_data["pr_case1"] = [orchard_anomaly_scores[orchard_id]["normal"][0] - x + orchard_anomaly_scores[orchard_id]["normal"][0] for x in orchard_data["pr_case1"]]
            for i in range(len(orchard_patch_results[orchard_id]["img"])):
                if orchard_patch_results[orchard_id]["label"][i] == 1:
                    orchard_patch_results[orchard_id]["score"][i] = orchard_anomaly_scores[orchard_id]["normal"][0] - orchard_patch_results[orchard_id]["score"][i] + orchard_anomaly_scores[orchard_id]["normal"][0]
        '''
        '''
        if (orchard_case_count[orchard_id]["case_2"] != 0 and 
        sum(orchard_data["pr_case2"]) / orchard_case_count[orchard_id]["case_2"] 
        < sum(orchard_data["pr_normal"]) / orchard_case_count[orchard_id]["normal"]):
            orchard_data["pr_case2"] = [orchard_anomaly_scores[orchard_id]["normal"][0] - x + orchard_anomaly_scores[orchard_id]["normal"][0] for x in orchard_data["pr_case2"]]
            for i in range(len(orchard_patch_results[orchard_id]["img"])):
                if orchard_patch_results[orchard_id]["label"][i] == 2:
                    orchard_patch_results[orchard_id]["score"][i] = orchard_anomaly_scores[orchard_id]["normal"][0] - orchard_patch_results[orchard_id]["score"][i] + orchard_anomaly_scores[orchard_id]["normal"][0]
        '''
        auroc_case1 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case1"])
        auroc_case2 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case2"]))], 
                                      orchard_data["pr_normal"] + orchard_data["pr_case2"])
        auroc_case3 = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case3"]))],
                                        orchard_data["pr_normal"] + orchard_data["pr_case3"])
        #auroc_total = calculate_auroc([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"] + orchard_data["pr_case2"]))], 
                                      #orchard_data["pr_normal"] + orchard_data["pr_case1"] + orchard_data["pr_case2"])
        auroc_total = sum([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None]) / len([x for x in [auroc_case1, auroc_case2, auroc_case3] if x is not None])
        print("[INFO] FINAL RESULTS:")
        print(f"- ID: {orchard_id}, CASE 1 AUROC: {auroc_case1}, CASE 2 AUROC: {auroc_case2}, CASE 3 AUROC: {auroc_case3} OVERALL: {auroc_total}")
        plot_histogram(orchard_data["pr_case1"], orchard_data["pr_case2"], orchard_data["pr_case3"], orchard_data["pr_normal"], orchard_id)

        fig, ax = plt.subplots()
        if auroc_case1:
            PrecisionRecallDisplay.from_predictions([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case1"]))],
                                                        orchard_data["pr_normal"] + orchard_data["pr_case1"], name="CASE 1 PRECISION RECALL", ax=ax)
        if auroc_case2:
            PrecisionRecallDisplay.from_predictions([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case2"]))],
                                                        orchard_data["pr_normal"] + orchard_data["pr_case2"], name="CASE 2 PRECISION RECALL", ax=ax)
        if auroc_case3:
            PrecisionRecallDisplay.from_predictions([0 for _ in range(len(orchard_data["pr_normal"]))] + [1 for _ in range(len(orchard_data["pr_case3"]))],
                                                        orchard_data["pr_normal"] + orchard_data["pr_case3"], name="CASE 3 PRECISION RECALL", ax=ax)
        plt.show()
        if n_plot_per_class > 0:        
            for i in range(len(orchard_patch_results[orchard_id]["img"])):
                plot_sample(orchard_patch_results[orchard_id]["img"][i], orchard_patch_results[orchard_id]["label"][i], orchard_patch_results[orchard_id]["score"][i], orchard_id)
            
def calculate_auroc(gt_list, pr_list):
    if len(set(gt_list)) == 2:
        auroc = round(roc_auc_score(gt_list, pr_list), 3)
        return auroc
    else:
        return None

'''
def test(_class_):
    """
        Load a trained model for a specific class and runs the evaluation.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '../mvtec/' + _class_
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_' + _class_ + '.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device,_class_)
    print(_class_,':',auroc_px,',',auroc_sp,',',aupro_px)
    return auroc_px
'''
import os

def visualization(_class_):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = '../mvtec/' + _class_
    ckp_path = './checkpoints/' + 'rm_1105_wres50_ff_mm_'+_class_+'.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])

    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            if (label.item() == 0):
                continue
            #if count <= 10:
            #    count += 1
            #    continue

            decoder.eval()
            bn.eval()

            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            #inputs.append(feature)
            #inputs.append(outputs)
            #t_sne(inputs)


            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            #if not os.path.exists('./results_all/'+_class_):
            #    os.makedirs('./results_all/'+_class_)
            #cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            #cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'ad.png', ano_map)
            plt.imshow(ano_map)
            plt.axis('off')
            #plt.savefig('ad.png')
            plt.show()

            gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            count += 1
            #if count>20:
            #    return 0
                #assert 1==2


def vis_nd(name, _class_):
    print(name,':',_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ckp_path = './checkpoints/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            #if count <= 10:
            #    count += 1
            #    continue
            img = img.to(device)
            inputs = encoder(img)
            #print(inputs[-1].shape)
            outputs = decoder(bn(inputs))


            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            #anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'org.png',img)
            #plt.imshow(img)
            #plt.axis('off')
            #plt.savefig('org.png')
            #plt.show()
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'ad.png', ano_map)
            #plt.imshow(ano_map)
            #plt.axis('off')
            #plt.savefig('ad.png')
            #plt.show()

            #gt = gt.cpu().numpy().astype(int)[0][0]*255
            #cv2.imwrite('./results/'+_class_+'_'+str(count)+'_'+'gt.png', gt)

            #b, c, h, w = inputs[2].shape
            #t_feat = F.normalize(inputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #s_feat = F.normalize(outputs[2], p=2).view(c, -1).permute(1, 0).cpu().numpy()
            #c = 1-min_max_norm(cv2.resize(anomaly_map,(h,w))).flatten()
            #print(c.shape)
            #t_sne([t_feat, s_feat], c)
            #assert 1 == 2

            #name = 0
            #for anomaly_map in amap_list:
            #    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            #    ano_map = min_max_norm(anomaly_map)
            #    ano_map = cvt2heatmap(ano_map * 255)
                #ano_map = show_cam_on_image(img, ano_map)
                #cv2.imwrite(str(name) + '.png', ano_map)
                #plt.imshow(ano_map)
                #plt.axis('off')
                #plt.savefig(str(name) + '.png')
                #plt.show()
            #    name+=1
            #count += 1
            #if count>40:
            #    return 0
                #assert 1==2
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))  # np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        ano_score = (prmean_list_sp-np.min(prmean_list_sp))/(np.max(prmean_list_sp)-np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)
        #print(type(vis_data))
        #np.save('vis.npy',vis_data)
        with open('vis.pkl','wb') as f:
            pickle.dump(vis_data,f,pickle.HIGHEST_PROTOCOL)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def detection(encoder, bn, decoder, dataloader,device,_class_):
    #_, t_bn = resnet50(pretrained=True)
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    #t_bn.to(device)
    #t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:

            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)


            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))#np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1


        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
    return auroc_sp_max, auroc_sp_mean
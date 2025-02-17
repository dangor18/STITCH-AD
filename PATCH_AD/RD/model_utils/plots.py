import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_channels(image, title):
    """
        Plot data channels when loading data, used for testing
    """
    channel_names = ['DEM', 'Edge', 'Red', 'Red spec', 'Red Edge', 'nir']
    # number of plots for each channel
    n = image.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(15, 5))
    fig.suptitle(title, size=20)
    for i in range(n):
        axs[i].set_xlabel('X', size=16)
        axs[i].set_ylabel('Y', size=16)
        axs[i].set_title(f'{channel_names[i]} Channel', size=18)
        axs[i].axis('off')
        if i == 0:
            plt.colorbar(axs[i].imshow(image[i].numpy(), cmap='viridis'), ax=axs[i], label='Value')
        else:
            plt.colorbar(axs[i].imshow(image[i].numpy(), cmap='gray'), ax=axs[i], label='Value')
    plt.tight_layout()
    plt.show()

def plot_sample(image, label, anomaly_score, orchard_id):
    """
        Plot each channel of an image, with label and anomaly score for a given orchard
    """
    if label == "normal":
        color = 'green'
    else:
        color = 'red'
    channel_names = ['DEM', 'Edge', 'Red', 'Red spec', 'Red Edge', 'nir']
    image = image.cpu().numpy().squeeze()
    # number of plots for each channel
    n = image.shape[0]
    fig, axs = plt.subplots(1, n+1, figsize=(24, 6))
    fig.suptitle(f"CLASS ID {orchard_id}", size=20)

    for i in range(n):
        axs[i].set_xlabel('X', size=16)
        axs[i].set_ylabel('Y', size=16)
        axs[i].set_title(f'{channel_names[i]} Channel', size=18, pad=10)
        axs[i].axis('off')

        channel_data = image[i]
        if i == 0:
            plt.colorbar(axs[i].imshow(channel_data, cmap='viridis'), ax=axs[i], label='Value')
        else:
            plt.colorbar(axs[i].imshow(channel_data, cmap='gray'), ax=axs[i], label='Value')
    
    axs[-1].axis('off')
    axs[-1].text(0.5, 0.6, f"LABEL: {label}", ha='center', va='center', fontsize=14)
    axs[-1].text(0.5, 0.4, f"ANOMALY SCORE: {anomaly_score:.4f}", ha='center', va='center', fontsize=18, color=color)

    plt.tight_layout()
    plt.show()

def plot_histogram(case_1_scores, case_2_scores, case_3_scores, normal_scores, orchard_id):
    """
    Plot histogram for score distribution
    ARGS: case 1 scores, case 2 scores, case 3 scores, normal scores, orchard id: str 
    """
    plt.figure(figsize=(12, 6))
    combined_scores = np.concatenate([normal_scores] + ([case_1_scores] if len(case_1_scores) > 0 else []) + ([case_2_scores] if len(case_2_scores) > 0 else [])  + ([case_3_scores] if len(case_3_scores) > 0 else []))
    
    min = np.min(combined_scores)
    max = np.max(combined_scores)
    bins = np.linspace(min, max, 100)        # 50 bins between min and max

    plt.hist(normal_scores, bins=bins, alpha=0.7, label='Normal', color='green')
    if len(case_1_scores) > 0:
        plt.hist(case_1_scores, bins=bins, alpha=0.7, label='Case 1', color='red')
    if len(case_2_scores) > 0:
        plt.hist(case_2_scores, bins=bins, alpha=0.7, label='Case 2', color='blue')
    if len(case_3_scores) > 0:
        plt.hist(case_3_scores, bins=bins, alpha=0.7, label='Case 3', color='orange')

    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title(f'Anomaly Scores for Orchard {orchard_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.axvline(np.mean(normal_scores), color='green', linestyle='dashed', linewidth=2)
    if len(case_1_scores) > 0:
        plt.axvline(np.mean(case_1_scores), color='red', linestyle='dashed', linewidth=2)
    if len(case_2_scores) > 0:
        plt.axvline(np.mean(case_2_scores), color='blue', linestyle='dashed', linewidth=2)
    if len(case_3_scores) > 0:
        plt.axvline(np.mean(case_3_scores), color='orange', linestyle='dashed', linewidth=2)

    plt.tight_layout()
    plt.show()

def plot_auroc(auroc_dict):
    """
        Plot the auroc dict accumalated during training for per orchard auroc at each epoch
    """
    plt.figure(figsize=(12, 6))
    x = auroc_dict.keys()
    temp_dict = {}
    for _, orchard_auroc_list in auroc_dict.items():
        y = []
        for orchard_id, auroc in orchard_auroc_list.items():
            if orchard_id not in temp_dict:
                temp_dict[orchard_id] = []
            temp_dict[orchard_id].append(auroc)
        
    for orchard_id, auroc_list in temp_dict.items():
        plt.plot(x, auroc_list, 'o-', label=f'Orchard {orchard_id}')

    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.title('AUROC per Orchard over Epochs')
    plt.legend()
    #plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def get_class_name(label):
    if label == "normal":
        return "Normal"
    elif label == "case_1":
        return "Case 1"
    elif label == "case_2":
        return "Case 2"
    elif label == "case_3":
        return "Case 3"
    else:
        return "Unknown"
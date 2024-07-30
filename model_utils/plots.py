import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_sample(image, label, anomaly_score, orchard_id):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    if not np.ndim(image) == 2:
        image = torch.squeeze(image)
    print(np.shape(image))
    plt.title(orchard_id)
    for i, channel_name in enumerate(['dem', 'ndvi', 'threshold']):
        img = image[i].cpu().numpy()
        axs[i].imshow(image[i].cpu().numpy(), cmap='gray')
        axs[i].set_title(f'{channel_name} Channel')
        axs[i].axis('off')
    
    axs[3].axis('off')
    axs[3].text(0.5, 0.6, f"LABEL: {get_class_name(label)}", ha='center', va='center', fontsize=12)
    axs[3].text(0.5, 0.4, f"ANOMALY SCORE: {anomaly_score:.4f}", ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_histogram(case_1_scores, case_2_scores, normal_scores, orchard_id):
    plt.figure(figsize=(12, 6))
    combined_scores = np.concatenate([normal_scores] + ([case_1_scores] if len(case_1_scores) > 0 else []) + ([case_2_scores] if len(case_2_scores) > 0 else []))
    
    min = np.min(combined_scores)
    max = np.max(combined_scores)
    bins = np.linspace(min, max, 100)        # 50 bins between min and max

    plt.hist(normal_scores, bins=bins, alpha=0.7, label='Normal', color='green')
    if len(case_1_scores) > 0:
        plt.hist(case_1_scores, bins=bins, alpha=0.7, label='Case 1', color='red')
    if len(case_2_scores) > 0:
        plt.hist(case_2_scores, bins=bins, alpha=0.7, label='Case 2', color='blue')

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

    plt.tight_layout()
    plt.show()

def plot_auroc(auroc_dict):
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
    plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def get_class_name(label):
    if label == 0:
        return "Normal"
    elif label == 1:
        return "Case 1"
    elif label == 2:
        return "Case 2"
    elif label == 3:
        return "Case 3"
    else:
        return "Unknown"
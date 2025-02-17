import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pathlib import Path
from itertools import product

def create_color_map(num_colors: int) -> np.ndarray:
    return plt.cm.tab20(np.linspace(0, 1, num_colors))[:, :3]

def overlay_colored_masks(image: np.ndarray, masks: list[dict], alpha: float = 0.5) -> np.ndarray:
    if not masks:
        return image.copy()
    overlay = image.copy().astype(np.float32)
    colors = create_color_map(len(masks))
    combined_mask = np.zeros_like(overlay)
    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        bool_mask = mask['segmentation']
        combined_mask[bool_mask] = color * 255
    overlay = cv2.addWeighted(overlay, 1.0, combined_mask, alpha, 0)
    return np.clip(overlay, 0, 255).astype(np.uint8)

def get_sam_configs() -> list[dict]:
    stability_scores = [0.80, 0.85, 0.90, 0.95]
    iou_thresholds = [0.05, 0.07, 0.09, 0.11]
    
    return [
        {
            'stability_score_thresh': stability,
            'pred_iou_thresh': iou,
            'points_per_side': 32,
            'box_nms_thresh': 0.7,
            'crop_n_layers': 1,
            'crop_nms_thresh': 0.7
        }
        for stability, iou in product(stability_scores, iou_thresholds)
    ]

def analyze_configs(image_path: str, output_dir: str):
    # Validate paths
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry['vit_b'](checkpoint='PRE_PROCESS/SEGMENTATION/sam_vit_b_01ec64.pth')
    sam.to(device=device)
    
    configs = get_sam_configs()
    n_configs = len(configs)
    n_cols = 4
    n_rows = int(np.ceil(n_configs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    print(configs)
    
    for idx, config in enumerate(configs):
        mask_generator = SamAutomaticMaskGenerator(sam, **config)
        masks = mask_generator.generate(img)
        seg_img = overlay_colored_masks(img, masks)
        
        axes[idx].imshow(seg_img)
        axes[idx].set_title(
            f'Stability: {config["stability_score_thresh"]}\n'
            f'IoU: {config["pred_iou_thresh"]}\n'
            f'Segments: {len(masks)}'
        )
        axes[idx].axis('off')
    
    # Remove empty subplots
    for idx in range(len(configs), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('SAM Configuration Comparison', fontsize=16, y=0.92)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'sam_config_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    INPUT_DIR = 'test_images'
    OUTPUT_DIR = 'test_output'
    TARGET_IMAGE = os.path.join(INPUT_DIR, 'prob_2.png')  # Update this path
    
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        analyze_configs(TARGET_IMAGE, OUTPUT_DIR)
        print(f"Analysis complete. Output saved to: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check that the image path is correct and the file exists.")
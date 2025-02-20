import os
import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import yaml
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from scipy import ndimage, stats
import concurrent.futures
from matplotlib import pyplot as plt
import random

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_blend_mask(height, width, overlap=32, device=None):
    """
    Create a blending mask for smooth transitions between chunks
    
    Args:
        height (int): Height of mask
        width (int): Width of mask 
        overlap (int): Overlap size in pixels
        device: torch device if using GPU
    """
    if device is not None:
        mask = torch.ones((height, width), device=device)
    else:
        mask = np.ones((height, width))
    
    for i in range(overlap):
        factor = i / overlap
        mask[:, i] *= factor
        mask[:, -(i+1)] *= factor
        mask[i, :] *= factor 
        mask[-(i+1), :] *= factor
    return mask

def read_and_resample_block(input_file, window_data, scale_factor, overlap):
    """
    Read and resample a block with proper blending mask
    """
    with rasterio.open(input_file) as src:
        col_off, row_off, width, height = window_data
        
        # Create expanded window with overlap
        expanded_window = Window(
            max(0, col_off - overlap),
            max(0, row_off - overlap), 
            min(src.width - col_off + overlap, width + 2*overlap),
            min(src.height - row_off + overlap, height + 2*overlap)
        )
        
        data = src.read(window=expanded_window)
        out_height = int(expanded_window.height * scale_factor)
        out_width = int(expanded_window.width * scale_factor)
        
        resampled = np.zeros((data.shape[0], out_height, out_width), dtype=np.float32)
        
        # Resample each band
        for i in range(data.shape[0]):
            resampled[i] = src.read(
                i + 1,
                out_shape=(out_height, out_width),
                window=expanded_window,
                resampling=Resampling.lanczos
            )
        
        # Create and apply blend mask
        blend_mask = create_blend_mask(out_height, out_width, int(overlap * scale_factor))
        resampled *= blend_mask
        
        # Calculate valid region
        resampled_overlap = int(overlap * scale_factor)
        start_row = resampled_overlap if row_off > 0 else 0 
        start_col = resampled_overlap if col_off > 0 else 0
        end_row = out_height - resampled_overlap if row_off + height < src.height else out_height
        end_col = out_width - resampled_overlap if col_off + width < src.width else out_width
        
        clipped = resampled[:, start_row:end_row, start_col:end_col]
        
        return clipped, (col_off, row_off, width, height)

def downscale_tif(input_file, config):
    """
    Downscale TIF using CPU or GPU based on availability
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return downscale_tif_gpu(input_file, config)
    else:
        return downscale_tif_cpu(input_file, config)

def downscale_tif_cpu(input_file, config):
    """
    CPU implementation of TIF downscaling
    """
    target_size = tuple(config['downscaling']['target_size'])
    chunk_size = config['downscaling']['chunk_size']
    overlap = config['downscaling'].get('overlap', 128)
    
    with rasterio.open(input_file) as src:
        scale_factor = min(target_size[0] / src.height, target_size[1] / src.width)
        output_height = int(src.height * scale_factor)
        output_width = int(src.width * scale_factor)
        
        output_profile = src.profile.copy()
        output_profile.update({
            'height': output_height,
            'width': output_width,
            'transform': src.transform * src.transform.scale(
                (src.width / output_width),
                (src.height / output_height)
            )
        })
        
        # Initialize output arrays
        final_sum = np.zeros((src.count, output_height, output_width), dtype=np.float32)
        weights_sum = np.zeros((output_height, output_width), dtype=np.float32)
        
        # Create processing windows
        windows = [
            (col, row, min(chunk_size, src.width - col), min(chunk_size, src.height - row))
            for row in range(0, src.height, chunk_size-2*overlap)
            for col in range(0, src.width, chunk_size-2*overlap)
        ]
        
        num_workers = min(config['downscaling']['num_workers'], 
                         multiprocessing.cpu_count())
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(read_and_resample_block, input_file, window, scale_factor, overlap) 
                for window in windows
            ]
            
            for future in futures:
                resampled, window_data = future.result()
                
                # Calculate output coordinates
                out_y = int(window_data[1] * scale_factor)
                out_x = int(window_data[0] * scale_factor)
                
                # Add to final arrays
                h, w = resampled.shape[1:]
                final_sum[:, out_y:out_y+h, out_x:out_x+w] += resampled
                weights_sum[out_y:out_y+h, out_x:out_x+w] += 1
        
        # Normalize and clip final image
        weights_sum = np.maximum(weights_sum, 1e-10)
        downscaled_image = np.clip(
            final_sum / weights_sum[np.newaxis, :, :],
            0, 255
        ).astype(np.uint8)
        
        return downscaled_image, output_profile

def downscale_tif_gpu(input_file, config):
    """
    GPU implementation of TIF downscaling
    """
    target_size = tuple(config['downscaling']['target_size'])
    overlap = config['downscaling'].get('overlap', 128)
    device = torch.device('cuda')
    
    with rasterio.open(input_file) as src:
        scale_factor = min(target_size[0] / src.height, target_size[1] / src.width)
        output_height = int(src.height * scale_factor)
        output_width = int(src.width * scale_factor)
        
        output_profile = src.profile.copy()
        output_profile.update({
            'height': output_height,
            'width': output_width,
            'transform': src.transform * src.transform.scale(
                (src.width / output_width),
                (src.height / output_height)
            )
        })
        
        # Calculate memory-efficient strip size
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = int(gpu_memory * 0.8)  # Use 80% of GPU memory
        bytes_per_pixel = 4  # float32
        row_memory = src.width * src.count * bytes_per_pixel
        max_rows = min(int(available_memory / row_memory), src.height)
        
        # Initialize output tensors
        final_sum = torch.zeros((src.count, output_height, output_width), 
                              dtype=torch.float32, device=device)
        weights_sum = torch.zeros((output_height, output_width), 
                                dtype=torch.float32, device=device)
        
        # Process image in vertical strips
        for y_start in range(0, src.height, max_rows - overlap):
            y_end = min(y_start + max_rows, src.height)
            
            # Read strip
            window = Window(0, y_start, src.width, y_end - y_start)
            strip_data = torch.from_numpy(src.read(window=window)).float().to(device)
            
            # Resample strip
            out_shape = (int((y_end - y_start) * scale_factor), 
                        int(src.width * scale_factor))
            resampled = F.interpolate(
                strip_data.unsqueeze(0),
                size=out_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Create and apply blend mask
            blend_mask = create_blend_mask(
                out_shape[0], out_shape[1],
                int(overlap * scale_factor), 
                device
            )
            
            # Add to output tensors
            out_y = int(y_start * scale_factor)
            final_sum[:, out_y:out_y + resampled.shape[1], :] += resampled * blend_mask
            weights_sum[out_y:out_y + resampled.shape[1], :] += blend_mask
            
            # Clean up GPU memory
            del strip_data, resampled
            torch.cuda.empty_cache()
        
        # Normalize and return final image
        weights_sum = torch.maximum(weights_sum, torch.tensor(1e-10, device=device))
        downscaled_image = torch.clip(
            final_sum / weights_sum,
            0, 255
        ).cpu().numpy().astype(np.uint8)
        
        return downscaled_image, output_profile

def load_sam_model(config):
    """
    Load the SAM (Segment Anything Model) model.

    Args:
        config (dict): Configuration dictionary containing SAM model parameters.

    Returns:
        SamAutomaticMaskGenerator: Initialized SAM mask generator.
    """
    # Set seeds for reproducibility 
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    sam_checkpoint = config['sam_checkpoint']
    model_type = config['model_type']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    if device == 'cuda':
        sam = sam.half()  # Enable FP16 for better efficiency
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=config['points_per_side'],
        pred_iou_thresh=config['pred_iou_thresh'],
        stability_score_thresh=config['stability_score_thresh'],
        box_nms_thresh=config['box_nms_thresh'],
        crop_nms_thresh=config['crop_nms_thresh'],
        crop_n_layers=config['crop_n_layers']
    )
    return mask_generator

def scale_for_sam(image, target_size):
    """
    Further downscale the image for SAM processing.

    Args:
        image (numpy.ndarray): Input image.
        target_size (tuple): Target size (height, width) for downscaling.

    Returns:
        numpy.ndarray: Downscaled image.
    """
    if image.dtype == bool:
        image = (image.astype(np.uint8) * 255)  # Convert binary mask to uint8

    h, w = image.shape[:2]
    target_width = target_size[1]  # Fixed width = 4096
    scale = target_width / w  # Compute scaling factor based on width
    new_height = max(1, int(h * scale))  # Ensure nonzero height
    new_size = (target_width, new_height)  # OpenCV expects (width, height)
    if target_size[0] != target_size[1]:
        new_size = target_size[::-1]
    
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)

def show_masks(image, masks, random_colors=True):
    """
    Visualize masks overlaid on the image.

    Args:
        image (numpy.ndarray): Original image.
        masks (list): List of mask dictionaries from SAM.
        random_colors (bool): Whether to use random colors for masks.
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    # Create a color map
    if random_colors:
        color_map = plt.cm.get_cmap('tab20')  # You can try other colormaps like 'Set1', 'Set2', 'Set3', etc.
    else:
        color_map = plt.cm.get_cmap('viridis')  # A sequential colormap
    
    # Create a single mask that combines all individual masks
    combined_mask = np.zeros(image.shape[:2] + (4,), dtype=np.float32)
    
    for i, mask in enumerate(masks):
        mask_image = mask['segmentation']
        if random_colors:
            color = color_map(random.random())
        else:
            color = color_map(i / len(masks))
        
        mask_color = np.concatenate([color[:3], [0.7]])  # RGBA
        combined_mask[mask_image] = mask_color
    
    plt.imshow(combined_mask)
    plt.title(f"Number of segments: {len(masks)}")
    plt.axis('off')
    plt.show()
    
def save_debug_image(image, filename="", output_dir="", id=""):
    """
    Save debug image in both PNG and NPY formats, handling different shapes appropriately.
    
    Args:
        image: Input image/mask that could be:
            - HWC format (height, width, channels)
            - CHW format (channels, height, width)
            - HW format (height, width) for single-channel masks
        filename: Name for the saved files
        output_dir: Output directory
    """
    # return
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    if filename != "":
        filename = filename[18:23]
    
    # Handle different input shapes
    if image.ndim == 2:  # Single channel mask
        filled_mask = (image > 0).astype(np.uint8) * 255
        plt.imsave(os.path.join(debug_dir, f"{filename+id}.png"), filled_mask, cmap='gray')
    elif image.ndim == 3:
        if image.shape[0] in [3, 4]:  # CHW format
            image_to_save = image.transpose(1, 2, 0)
        else:  # Already in HWC format
            image_to_save = image
        plt.imsave(os.path.join(debug_dir, f"{filename + id}.png"), image_to_save) 

def apply_autumn_filter(image):
    """Convert RGB image to autumn colors while preserving alpha channel"""
    
    if image.shape[2] > 4:
        image = image.transpose(1, 2, 0)
        print(f'Image shape after transpose: {image.shape}')
    
    # Extract alpha and RGB channels
    alpha = image[:, :, 3]  # Get alpha channel
    rgb = image[:, :, :3]   # Get RGB channels
    
    # Process RGB channels
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    autumn_hsv = hsv.copy()
    autumn_hsv[green_mask > 0] = np.array([15, 200, 200])
    variation = np.random.randint(-10, 10, autumn_hsv.shape)
    autumn_hsv = np.clip(autumn_hsv + variation, 0, 255).astype(np.uint8)
    
    autumn_rgb = cv2.cvtColor(autumn_hsv, cv2.COLOR_HSV2RGB)
    rgb_result = cv2.addWeighted(rgb, 0.3, autumn_rgb, 0.7, 0)
    
    # Recombine with alpha channel
    result = np.dstack([rgb_result, alpha])
    
    print("Output image shape:", result.shape)
    # Convert to channel-first format
    return result.transpose(2, 0, 1)
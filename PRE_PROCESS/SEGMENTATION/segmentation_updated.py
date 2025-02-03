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

def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
    sam_checkpoint = config['segmentation']['sam_checkpoint']
    model_type = config['segmentation']['model_type']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    if device == 'cuda':
        sam = sam.half()  # Enable FP16 for better efficiency
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        pred_iou_thresh=config['segmentation']['pred_iou_thresh'],
        stability_score_thresh=config['segmentation']['stability_score_thresh'],
        box_nms_thresh=config['segmentation']['box_nms_thresh'],
        crop_nms_thresh=config['segmentation']['crop_nms_thresh']
    )
    return mask_generator

def create_threshold_mask(image, config):
    """Apply preprocessing with enhanced normalization"""
    image = image.transpose(1, 2, 0)
    img_f = image.astype(np.float32)
    
    # Calculate vegetation indices
    ratio = 255 * (img_f[:,:,1] / (img_f[:,:,0] + 1))
    shadow = np.log1p(img_f[:,:,1]) - np.log1p(img_f[:,:,2]) 
    seasonal = (img_f[:,:,1] / (img_f[:,:,0] + img_f[:,:,1] + img_f[:,:,2] + 1)) * 255

    # Normalize each channel individually
    ratio = cv2.normalize(ratio, None, 0, 255, cv2.NORM_MINMAX)
    shadow = cv2.normalize(shadow * 85, None, 0, 255, cv2.NORM_MINMAX)
    seasonal = cv2.normalize(seasonal, None, 0, 255, cv2.NORM_MINMAX)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ratio = clahe.apply(ratio.astype(np.uint8))
    shadow = clahe.apply(shadow.astype(np.uint8))
    seasonal = clahe.apply(seasonal.astype(np.uint8))

    # Combine channels with orchard areas enhanced
    result = np.dstack([ratio, shadow, seasonal])
    
    # Additional normalization to ensure orchards are lighter
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    
    return np.uint8(result)

def further_downscale_for_sam(image, target_size):
    """
    Further downscale the image for SAM processing.

    Args:
        image (numpy.ndarray): Input image.
        target_size (tuple): Target size (height, width) for downscaling.

    Returns:
        numpy.ndarray: Downscaled image.
    """
    h, w = image.shape[:2]
    max_size = target_size[0]
    
    if max(h, w) > max_size:
        scale = max_size / max(h, w) 
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size)
    return image

def remove_small_segments(masks, min_size, image_shape, image, pixel_value_threshold):
    """
    Remove small segments from multiple masks and combine them.

    Args:
        masks (list): List of mask dictionaries from SAM.
        min_size (float): Minimum size threshold for segments.
        image_shape (tuple): Shape of the original image.
        image (numpy.ndarray): Original image.
        pixel_value_threshold (float): Threshold for average pixel value.

    Returns:
        numpy.ndarray: Combined binary mask with small segments removed.
    """
    combined_mask = np.zeros(image_shape, dtype=bool)
    total_pixels = image_shape[0] * image_shape[1]
    for mask in masks:
        segment = mask['segmentation']
        if (np.sum(segment)/total_pixels >= min_size) or (np.sum(segment) >= 12000):
            segmented_pixels = image[segment]
            avg_pixel_value = np.mean(segmented_pixels)/255
            # Check if the segment is a dam
            num_pixels = np.sum(segment)
            if (avg_pixel_value >= pixel_value_threshold) or num_pixels/total_pixels >= 0.3:
                # corrode the mask
                kernel = np.ones((3, 3), np.uint8)
                segment = cv2.erode(segment.astype(np.uint8), kernel, iterations=1).astype(bool)
                combined_mask = np.logical_or(combined_mask, segment)         
    return combined_mask

def process_segment(args):
    """
    Process a single segment for nodata removal.

    Args:
        args (tuple): (label, segment, nodata_mask, max_nodata_percentage, kernel)

    Returns:
        int or None: Segment label if it should be removed, None otherwise.
    """
    label, segment, nodata_mask, max_nodata_percentage, kernel = args
    if np.packbits(segment).sum() * 8 < 20000:  # Approximate but faster count
        return label
    kernel = kernel.astype(np.uint8)
    dilated_segment = cv2.dilate(segment.astype(np.uint8), kernel, iterations=1)
    nodata_overlap = cv2.bitwise_and(dilated_segment, nodata_mask.astype(np.uint8))
    nodata_count = np.unpackbits(np.packbits(nodata_overlap)).sum()
    segment_size = np.unpackbits(np.packbits(dilated_segment)).sum()
    if (segment_size > 0 and (nodata_count / segment_size) > max_nodata_percentage):
        return label
    return None

def remove_nodata_segments(mask, rgb_image, nodata_value, max_nodata_percentage, border_size, num_threads=4):
    """
    Remove segments with a high percentage of no-data pixels.

    Args:
        mask (numpy.ndarray): Input binary mask.
        rgb_image (numpy.ndarray): Original RGB image.
        nodata_value (numpy.ndarray): Value representing no-data pixels.
        max_nodata_percentage (float): Maximum allowed percentage of no-data pixels.
        border_size (int): Size of the border for dilation.
        num_threads (int): Number of threads for parallel processing.

    Returns:
        numpy.ndarray: Updated mask with high no-data segments removed.
    """
    labeled, num_features = ndimage.label(mask)
    if rgb_image.shape[0] == 4:
        rgb_image = rgb_image.transpose(1, 2, 0)
    
    # Create nodata mask
    nodata_mask = np.all(rgb_image == nodata_value, axis=-1)
    
    # Create morphological kernel
    kernel = np.ones((border_size, border_size), dtype=bool)
        
    # Prepare arguments for multiprocessing
    segments = [labeled == i for i in range(1, num_features + 1)]
    args = [(i+1, segment, nodata_mask, max_nodata_percentage, kernel) for i, segment in enumerate(segments)]
    
    # Process segments in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_segment, args))
    
    # Remove segments that exceed the nodata threshold
    segments_to_remove = [label for label in results if label is not None]
    mask[np.isin(labeled, segments_to_remove)] = 0
    
    return mask

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
        
        mask_color = np.concatenate([color[:3], [1.0]])  # RGBA
        combined_mask[mask_image] = mask_color
    
    plt.imshow(combined_mask)
    plt.title(f"Number of segments: {len(masks)}")
    plt.axis('off')
    plt.show()

def segment_image(image, mask_generator, config):
    """
    Apply the SAM model to the image and process the resulting masks.

    Args:
        image (numpy.ndarray): Input image.
        mask_generator (SamAutomaticMaskGenerator): SAM mask generator.
        config (dict): Configuration dictionary.

    Returns:
        numpy.ndarray: Combined binary mask after processing.
    """ 
    # Ensure image is in the correct format for SAM (3D, RGB)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    
    # Generate masks
    with torch.amp.autocast('cuda'):
        masks = mask_generator.generate(image)
    # show_masks(image, masks, random_colors=True)
    # Remove small segments and combine masks
    min_segment_size = config['segmentation']['min_segment_size']
    combined_mask = remove_small_segments(masks, min_segment_size, image.shape[:2], image, config['segmentation']['pixel_value_threshold'])
    return combined_mask

def keep_largest_segment(mask):
    """
    Keep only the largest connected segment in the mask.

    Args:
        mask (numpy.ndarray): Input binary mask.

    Returns:
        numpy.ndarray: Mask containing only the largest segment.
    """
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        largest_mask = (labeled == largest_label)
        return largest_mask
    else:
        return mask

def detect_and_remove_dams(mask, image, outlier_threshold=1.5):
    """
    Detect potential dams and remove them from the mask.

    Args:
        mask (numpy.ndarray): Input binary mask.
        image (numpy.ndarray): Original image.
        outlier_threshold (float): Z-score threshold for outlier detection.

    Returns:
        numpy.ndarray: Updated mask with potential dams removed.
    """
    labeled_mask, num_features = ndimage.label(mask)
    
    # Handle channel-first format
    if len(image.shape) == 3:
        if image.shape[0] in [3, 4]:  # If channels first
            image = image.transpose(1, 2, 0)  # Convert to HWC
            
    # Skip if no segments found
    if num_features == 0:
        return mask
    
    segment_stats = []
    for i in range(1, num_features + 1):
        segment = (labeled_mask == i)
        segment_pixels = image[segment]
        if len(segment_pixels) > 0:
            std_rgb = np.std(segment_pixels[:, :3], axis=0)  # Only use RGB channels
            segment_stats.append(std_rgb)
    
    segment_stats = np.array(segment_stats)
    z_scores = stats.zscore(segment_stats, axis=0)
    outliers = np.any(z_scores < -outlier_threshold, axis=1)
    for i, is_outlier in enumerate(outliers):
        if is_outlier:
            mask[labeled_mask == (i + 1)] = False
    return mask

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
    
    cv2.imwrite("autumn_image.png", cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
    print("Output image shape:", result.shape)
    # Convert to channel-first format
    return result.transpose(2, 0, 1)

def save_debug_image(image, filename, output_dir):
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
    return
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Handle different input shapes
    if image.ndim == 2:  # Single channel mask
        plt.imsave(os.path.join(debug_dir, f"{filename}.png"), image, cmap='gray')
    elif image.ndim == 3:
        if image.shape[0] in [3, 4]:  # CHW format
            image_to_save = image.transpose(1, 2, 0)
        else:  # Already in HWC format
            image_to_save = image
        plt.imsave(os.path.join(debug_dir, f"{filename}.png"), image_to_save)
    
    
def process_single_file(input_file, output_dir, config, mask_generator):
    """
    Process a single input file with debug image saves at key pipeline stages.

    Args:
        input_file (str): Path to the input file.
        output_dir (str): Directory to save output files.
        config (dict): Configuration dictionary.
        mask_generator (SamAutomaticMaskGenerator): SAM mask generator.

    Returns:
        tuple: (success, downscaled_image, output_profile)
            success (bool): Whether processing was successful.
            downscaled_image (numpy.ndarray): Downscaled image data.
            output_profile (dict): Rasterio profile for the output.
    """
    print(f"Processing file: {input_file}")
    rel_path = os.path.relpath(input_file, config['root_dir'])
    unique_id = rel_path.replace(os.path.sep, '_').replace('.', '_')
    
    downscaled_file = os.path.join(output_dir, "orthos", f"downscaled_{unique_id}.tif")
    mask_file = os.path.join(output_dir, "masks", f"seg_mask_{unique_id}.tif")

    os.makedirs(os.path.dirname(downscaled_file), exist_ok=True)
    os.makedirs(os.path.dirname(mask_file), exist_ok=True)

    # Downscaling and loading initial image
    if os.path.exists(downscaled_file):
        print("Loading existing downscaled image...")
        with rasterio.open(downscaled_file) as src:
            downscaled_image = src.read()
            output_profile = src.profile.copy()
    else:
        downscaled_image, output_profile = downscale_tif(input_file, config)
        print("Saving downscaled image...")
        with rasterio.open(downscaled_file, 'w', **output_profile) as dst:
            dst.write(downscaled_image)
    
    save_debug_image(downscaled_image, f"{unique_id}_01_downscaled", output_dir)

    original_image = downscaled_image
    save_debug_image(original_image, f"{unique_id}_02_original", output_dir)
    
    # Apply autumn filter
    image = original_image
    # image = apply_autumn_filter(original_image)
    save_debug_image(image, f"{unique_id}_03_autumn", output_dir)
    # print(f'SHape of autum image: {image.shape}')
    
    # Threshold mask creation
    threshold_mask = create_threshold_mask(image, config)
    save_debug_image(threshold_mask, f"{unique_id}_04_threshold", output_dir)
    
    # SAM preprocessing
    sam_image = further_downscale_for_sam(threshold_mask, config['segmentation']['sam_target_size'])
    save_debug_image(sam_image, f"{unique_id}_05_sam_input", output_dir)

    # Segmentation
    segmentation_mask = segment_image(sam_image, mask_generator, config)
    save_debug_image(segmentation_mask.astype(np.uint8)*255, f"{unique_id}_06_segmentation", output_dir)
    
    print(segmentation_mask.shape)

    print("Segmentation completed.")
    
    print(f'Image shape before upscale: {image.shape}')
    
    # Mask upscaling
    upscaled_mask = cv2.resize(segmentation_mask.astype(np.uint8), (image.shape[2], image.shape[1]), 
                              interpolation=cv2.INTER_NEAREST)
    save_debug_image(upscaled_mask*255, f"{unique_id}_07_upscaled", output_dir)
    
    print("Upscaling completed.")

    # No-data removal
    nodata_value = np.array(config['segmentation']['nodata_value'])    
    max_nodata_percentage = config['segmentation']['max_nodata_percentage']
    border_size = config['segmentation'].get('border_size', 3)
    
    print("Removing no-data segments...")
    cleaned_mask = remove_nodata_segments(upscaled_mask, original_image, nodata_value, 
                                        max_nodata_percentage, border_size, num_threads=12)
    
    save_debug_image(cleaned_mask*255, f"{unique_id}_08_cleaned", output_dir)
    
    print("No-data removal completed.")
    
    # Dam removal
    outlier_threshold = config['segmentation']['outlier_threshold']
    final_mask = detect_and_remove_dams(cleaned_mask, image, outlier_threshold)
    save_debug_image(final_mask*255, f"{unique_id}_09_no_dams", output_dir)
    
    print("Dam removal completed.")
    
    # Final mask processing
    final_mask = keep_largest_segment(np.logical_not(final_mask))
    final_mask = np.where(final_mask == 1, 0, -999)

    # Save final mask
    mask_profile = output_profile.copy()
    mask_profile.update(dtype=rasterio.float32, count=1, nodata=-999)
    with rasterio.open(mask_file, 'w', **mask_profile) as dst:
        dst.write(final_mask.astype(rasterio.float32), 1)

    return True, downscaled_image, output_profile

def main(config_file, root_dir):
    """
    Main function to process all files.

    Args:
        config_file (str): Path to the configuration file.
        root_dir (str): Root directory to search for input files.
    """
    config = load_config(config_file)
    config['root_dir'] = root_dir
    target_filename = config.get('input', {}).get('target_filename', 'orthomosaic_visible.tif')

    # Load the SAM model
    mask_generator = load_sam_model(config)

    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_filename in filenames:
            matching_files.append(os.path.join(dirpath, target_filename))

    output_dir = os.path.join(os.path.dirname(root_dir), "segmentation")
    os.makedirs(os.path.join(output_dir, "orthos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    for input_file in matching_files:
        # try:
        success, _, _ = process_single_file(input_file, output_dir, config, mask_generator)
        if success:
            print(f"Successfully processed {input_file}")
        else:
            print(f"Failed to process {input_file}")
        # except Exception as e:
        #     print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchard Downscaling and Segmentation")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("root_dir", type=str, help="Root directory to search for files")
    args = parser.parse_args()
    main(args.config, args.root_dir)
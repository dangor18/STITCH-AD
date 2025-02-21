# Standard library
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

# Third-party libraries
import cv2
import numpy as np
import rasterio
import yaml
import torch
from scipy import ndimage, stats

# Local imports
from segmentation_utils import (
    scale_for_sam, 
    downscale_tif, 
    load_sam_model, 
    show_masks, 
    apply_autumn_filter, 
    save_debug_image
)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def create_threshold_mask(image: np.ndarray, output_dir: str) -> np.ndarray:
    """Create threshold mask using edge detection and watershed segmentation."""
    # Convert to float32 for calculations
    img = image.astype(np.float32)
    
    # Calculate vegetation indices
    shadow = np.log1p(img[:,:,1]) - np.log1p(img[:,:,2])
    seasonal = (img[:,:,1] / (img[:,:,0] + img[:,:,1] + img[:,:,2] + 1)) * 255
    
    # Enhance using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    shadow = clahe.apply(cv2.normalize(shadow * 85, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    seasonal = clahe.apply(cv2.normalize(seasonal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    
    # Combine channels
    weighted = (shadow * 0.5 + seasonal * 0.5).astype(np.uint8)
    
    # Edge detection and enhancement
    blur = cv2.GaussianBlur(weighted, (5, 5), 0)
    sobel = np.sqrt(
        cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)**2 + 
        cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)**2
    )
    sobel = clahe.apply(cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    
    # Enhanced edge targeting
    enhanced = sobel.astype(np.float32)
    diff = enhanced - 10  # Target edge strength
    mask = np.abs(diff) < 10  # Delta range
    enhanced[mask] = 10 + (diff[mask] * 1.0)  # Apply strength multiplier
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    
    # Watershed segmentation
    markers = cv2.connectedComponents(
        cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    )[1] + 1
    markers[enhanced < 30] = 0
    
    watershed_mask = (cv2.watershed(
        cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), 
        markers
    ) > 1).astype(np.uint8) * 255
    
    save_debug_image(watershed_mask, output_dir=output_dir, id="03a_watershed")
    return watershed_mask

def remove_small_segments(masks: list, image_shape: tuple, image: np.ndarray, 
                        min_size: float, threshold: float) -> np.ndarray:
    """Filter and combine valid segments based on size ratios."""
    total_pixels = image_shape[0] * image_shape[1]
    valid_segments = []
    
    # Ensure image is in correct format (HWC)
    if len(image.shape) == 3 and image.shape[0] in [3, 4]:
        image = image.transpose(1, 2, 0)
    
    for mask in masks[1:]:  # Skip first mask as it's often background
        segment = mask['segmentation']
        # Resize segment if dimensions don't match
        if segment.shape != image.shape[:2]:
            segment = cv2.resize(
                segment.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
        segment_ratio = np.sum(segment) / total_pixels
        
        # Only check size ratios
        if (segment_ratio >= min_size) or (0.8 >= segment_ratio >= min_size):
            valid_segments.append(segment)
    
    # Combine valid segments with erosion
    kernel = np.ones((5, 5), np.uint8)
    combined = np.zeros(image.shape[:2], dtype=bool)
    print(f'Number of valid segments: {len(valid_segments)}')
    for segment in valid_segments:
        eroded = cv2.erode(segment.astype(np.uint8), kernel, iterations=2)
        combined = np.logical_or(combined, eroded.astype(bool))
    
    return combined

def process_segment(args):
    """Process a single segment checking only for nodata pixels."""
    label, segment, alpha_mask, max_percentage = args
    
    # Check alpha channel for nodata (0 values)
    segment_alpha_zeros = np.sum(segment & (alpha_mask == 0))
    segment_size = np.sum(segment)
    
    if segment_size > 0 and (segment_alpha_zeros / segment_size) > max_percentage:
        return label
        
    return None

def remove_nodata_segments(mask: np.ndarray, image: np.ndarray, **kwargs) -> np.ndarray:
    """Remove segments with excessive no-data pixels."""
    max_nodata = kwargs.get('max_nodata_percentage', 0.3)
    threads = kwargs.get('num_threads', 4)
    
    # Ensure image is in correct format
    if len(image.shape) == 3 and image.shape[0] in [3, 4]:
        image = image.transpose(1, 2, 0)
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    # Get alpha channel (assume it's the last channel)
    alpha_channel = image[:, :, -1] if image.shape[-1] == 4 else np.ones_like(mask) * 255
    
    # Create list of segments
    segments = [labeled == i for i in range(1, num_features + 1)]
    
    args = [(i+1, seg, alpha_channel, max_nodata) 
            for i, seg in enumerate(segments)]
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(process_segment, args))
    
    # Remove invalid segments
    mask[np.isin(labeled, [r for r in results if r is not None])] = 0
    return mask

def segment_image(image: np.ndarray, mask_generator, **kwargs) -> list:
    """Generate masks using SAM.
    
    Returns:
        list: List of mask dictionaries from SAM
    """
    target_size = kwargs.get('target_size', (1024, 1024))
    show = kwargs.get('show_masks', False)
    
    # Ensure correct format and size
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    original_size = image.shape[:2]
    image = scale_for_sam(image, target_size)
    
    # Generate masks
    with torch.amp.autocast('cuda'):
        masks = mask_generator.generate(image)
    
    if show:
        show_masks(image, masks, random_colors=True)
    
    return masks

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

def remove_outliers(mask: np.ndarray, image: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    """Remove segments with anomalous RGB statistics."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
        
    image = image.transpose(1, 2, 0) if len(image.shape) == 3 and image.shape[0] in [3, 4] else image
    
    # Calculate segment statistics
    stats_list = []
    for i in range(1, num_features + 1):
        segment = labeled == i
        pixels = image[segment]
        if len(pixels) > 0:
            stats_list.append(np.std(pixels[:, :3], axis=0))
    
    if not stats_list:
        return mask
        
    # Remove outlier segments
    outliers = np.any(stats.zscore(np.array(stats_list), axis=0) < -threshold, axis=1)
    for i, is_outlier in enumerate(outliers):
        if is_outlier:
            mask[labeled == (i + 1)] = False
    
    return mask

def extract_uog_id(filepath):
    """Extract UOG ID from filepath or generate a sequential one if not found."""
    import re
    
    # Try to find UOG_XXXX pattern in the filepath
    match = re.search(r'UOG_(\d{4})', filepath)
    if match:
        return match.group(0)
    
    # If not found, extract just the numbers
    numbers = re.findall(r'\d+', filepath)
    if numbers:
        # Use the last sequence of numbers found, zero-padded to 4 digits
        return f"UOG_{numbers[-1]:0>4}"
    
    # If no numbers found, return None
    return None

def process_single_file(input_file, output_dir, config, mask_generator, preprocess=True):
    """Process a single input file with progress tracking and timing information."""
    start_time = time.time()
    
    # Extract UOG ID
    uog_id = extract_uog_id(input_file)
    if not uog_id:
        print(f"Warning: Could not extract UOG ID from {input_file}")
        uog_id = "UOG_0000"
    
    filename = os.path.basename(input_file)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing: {uog_id} - {filename}")
    
    # Check if we should skip segmentation
    if config.get('skip_enabled', False) and input(f"Skip segmentation for {uog_id}? (y/n): ").lower() == 'y':
        print("→ Skipping segmentation")
        return False, None, None
    
    # Setup file paths
    rel_path = os.path.relpath(input_file, config['root_dir'])
    unique_id = f"{uog_id}_{rel_path.replace(os.path.sep, '_').replace('.', '_')}"
    downscaled_file = os.path.join(output_dir, "orthos", f"downscaled_{unique_id}.tif")
    mask_file = os.path.join(output_dir, "masks", f"seg_mask_{unique_id}.tif")

    os.makedirs(os.path.dirname(downscaled_file), exist_ok=True)
    os.makedirs(os.path.dirname(mask_file), exist_ok=True)

    # Downscaling phase
    if os.path.exists(downscaled_file):
        print("→ Loading existing downscaled image")
        with rasterio.open(downscaled_file) as src:
            downscaled_image = src.read()
            output_profile = src.profile.copy()
    else:
        print("→ Downscaling image...")
        downscale_start = time.time()
        downscaled_image, output_profile = downscale_tif(input_file, config)
        print(f"  ✓ Completed in {time.time() - downscale_start:.1f}s")
        
        with rasterio.open(downscaled_file, 'w', **output_profile) as dst:
            dst.write(downscaled_image)
    
    # Save debug image of downscaled result
    if config.get('save_debug', False):
        save_debug_image(
            downscaled_image.transpose(1, 2, 0), 
            output_dir=output_dir,
            filename=uog_id,
            id="_01_downscaled"
        )
    
    # Preprocessing phase
    image = downscaled_image
    if config.get('autumn_filtered', False):
        print(f"→ Applying autumn filter to {uog_id}...")
        image = apply_autumn_filter(image)
        if config.get('save_debug', False):
            save_debug_image(
                image.transpose(1, 2, 0), 
                output_dir=output_dir,
                filename=uog_id,
                id="_02_autumn"
            )
    
    config["output_dir"] = output_dir
    image = image.transpose(1, 2, 0)
    
    # Segmentation phase
    print("→ Running segmentation...")
    seg_start = time.time()
    
    if preprocess:
        sam_image = create_threshold_mask(image, output_dir)
    else:
        sam_image = image[:, :, :3] if image.shape[2] == 4 else image
    
    # Get SAM masks
    masks = segment_image(
        image=sam_image,
        mask_generator=mask_generator,
        target_size=config['segmentation']['sam_target_size'],
        show_masks=config.get('show_masks', False)
    )
    
    print(f'Number of masks: len(masks) = {len(masks)}')
    
    # Process and combine masks
    print("→ Processing segments...")
    combined_mask = remove_small_segments(
        masks, 
        sam_image.shape[:2], 
        sam_image,
        config['segmentation']['min_segment_size'],
        config['segmentation']['pixel_value_threshold']
    )
    
    if config.get('save_debug', False):
        save_debug_image(
            combined_mask,
            output_dir=output_dir,
            filename=uog_id,
            id="_03_initial_mask"
        )
    
    print(f"  ✓ Segmentation completed in {time.time() - seg_start:.1f}s")
    
    # Post-processing phase
    print("\n→ Starting post-processing pipeline...")
    post_start = time.time()
    
    print("  → Removing nodata segments...")
    nodata_start = time.time()
    cleaned_mask = remove_nodata_segments(
        mask=combined_mask,
        image=image,
        max_nodata_percentage=config['segmentation']['max_nodata_percentage'],
        num_threads=12
    )
    print(f"    ✓ Completed in {time.time() - nodata_start:.1f}s")
    
    if config.get('save_debug', False):
        save_debug_image(
            cleaned_mask,
            output_dir=output_dir,
            filename=uog_id,
            id="_04_cleaned_mask"
        )
    
    print("  → Removing statistical outliers...")
    outlier_start = time.time()
    final_mask = remove_outliers(cleaned_mask, image, config['segmentation']['outlier_threshold'])
    print(f"    ✓ Completed in {time.time() - outlier_start:.1f}s")
    
    if config.get('save_debug', False):
        save_debug_image(
            final_mask,
            output_dir=output_dir,
            filename=uog_id,
            id="_05_final_mask"
        )
    
    # Validation check
    print("  → Validating segmentation coverage...")
    validation_start = time.time()
    area_ratio = np.sum(final_mask) / (255 * final_mask.shape[0] * final_mask.shape[1])
    if area_ratio < 0.15:
        print(f"    ✗ Insufficient segmented area: {area_ratio:.1%}")
        return False, image, output_profile
    print(f"    ✓ Completed in {time.time() - validation_start:.1f}s")
    
    # Final processing
    print("  → Extracting largest segment...")
    final_start = time.time()
    final_mask = keep_largest_segment(np.logical_not(final_mask))
    final_mask = np.where(final_mask == 1, 0, -999)
    print(f"    ✓ Completed in {time.time() - final_start:.1f}s")
    
    total_post = time.time() - post_start
    print(f"✓ Post-processing completed in {total_post:.1f}s")
    
    # Save results
    print("\n→ Saving results...")
    save_start = time.time()
    mask_profile = output_profile.copy()
    mask_profile.update(dtype=rasterio.float32, count=1, nodata=-999)
    with rasterio.open(mask_file, 'w', **mask_profile) as dst:
        dst.write(final_mask.astype(rasterio.float32), 1)
    print(f"  ✓ Completed in {time.time() - save_start:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n✓ Total processing completed in {total_time:.1f}s")
    return True, image, output_profile

def print_config_summary(config):
    """Print a formatted summary of important configuration settings."""
    print("\n=== Configuration Summary ===")
    
    # SAM Model Settings
    print("\nSAM Model Settings:")
    print("------------------")
    for key in ['model_type', 'points_per_side', 'pred_iou_thresh', 
                'stability_score_thresh', 'box_nms_thresh', 'crop_nms_thresh',
                'crop_n_layers']:
        thresh_val = config['segmentation']['thresholded_settings'].get(key)
        unproc_val = config['segmentation']['unprocessed_settings'].get(key)
        if thresh_val != unproc_val:
            print(f"{key:.<25} Threshold: {thresh_val}, Unprocessed: {unproc_val}")
        else:
            print(f"{key:.<25} {thresh_val}")
    
    # Segmentation Settings
    print("\nSegmentation Settings:")
    print("--------------------")
    seg_config = config['segmentation']
    for key in ['sam_target_size', 'min_segment_size', 'pixel_value_threshold',
                'max_nodata_percentage', 'outlier_threshold']:
        print(f"{key:.<25} {seg_config.get(key)}")
    
    # Downscaling Settings
    print("\nDownscaling Settings:")
    print("-------------------")
    down_config = config['downscaling']
    for key in ['target_size', 'chunk_size', 'overlap']:
        print(f"{key:.<25} {down_config.get(key)}")
    
    # Processing Settings
    print("\nProcessing Settings:")
    print("------------------")
    print(f"{'Save debug images':.<25} {config.get('save_debug', False)}")
    print(f"{'Apply autumn filter':.<25} {config.get('autumn_filtered', False)}")
    print(f"{'Show masks':.<25} {config.get('show_masks', False)}")
    
    print("\n" + "="*50 + "\n")

def main(config_file, root_dir):
    start_time = time.time()
    print(f"\n=== Starting processing at {datetime.now().strftime('%H:%M:%S')} ===")
    
    config = load_config(config_file)
    config['root_dir'] = root_dir
    print_config_summary(config)
    target_filename = config.get('input', {}).get('target_filename', 'orthomosaic_visible.tif')

    # Load the SAM model
    mask_generator = load_sam_model(config['segmentation']['thresholded_settings'])

    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_filename in filenames:
            matching_files.append(os.path.join(dirpath, target_filename))

    output_dir = os.path.join(os.path.dirname(root_dir), "segmentation")
    os.makedirs(os.path.join(output_dir, "orthos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    print(f"Found {len(matching_files)} files to process")
    
    reprocess = []
    for i, input_file in enumerate(matching_files, 1):
        print(f"\nProcessing file {i}/{len(matching_files)}")
        success, _, _ = process_single_file(input_file, output_dir, config, mask_generator, preprocess=True)
        if not success:
            reprocess.append(input_file)
    
    if reprocess:
        print(f"\nRetrying {len(reprocess)} failed files with alternative settings...")
        mask_generator = load_sam_model(config['segmentation']['unprocessed_settings'])
        
        for i, input_file in enumerate(reprocess, 1):
            print(f"\nReprocessing file {i}/{len(reprocess)}")
            success, _, _ = process_single_file(input_file, output_dir, config, mask_generator, preprocess=False)
    
    total_time = time.time() - start_time
    print(f"\n=== Processing completed in {total_time:.1f}s ===")
    print(f"Successfully processed: {len(matching_files) - len(reprocess)} files")
    if reprocess:
        print(f"Failed to process: {len(reprocess)} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchard Downscaling and Segmentation")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("root_dir", type=str, help="Root directory to search for files")
    args = parser.parse_args()
    main(args.config, args.root_dir)

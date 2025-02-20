import os
import cv2
import numpy as np
import rasterio
import yaml
import argparse
import torch
from scipy import ndimage, stats
import concurrent.futures
from segmentation_utils import scale_for_sam, downscale_tif, load_sam_model, show_masks, apply_autumn_filter, save_debug_image

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

def create_threshold_mask(image, config):
    """Edge detection with targeted edge enhancement and watershed segmentation"""
    img_f = image.astype(np.float32)
    
    # Calculate vegetation indices
    shadow = np.log1p(img_f[:,:,1]) - np.log1p(img_f[:,:,2]) 
    seasonal = (img_f[:,:,1] / (img_f[:,:,0] + img_f[:,:,1] + img_f[:,:,2] + 1)) * 255

    # Initial normalization and enhancement
    shadow = cv2.normalize(shadow * 85, None, 0, 255, cv2.NORM_MINMAX)
    seasonal = cv2.normalize(seasonal, None, 0, 255, cv2.NORM_MINMAX)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    shadow = clahe.apply(shadow.astype(np.uint8))
    seasonal = clahe.apply(seasonal.astype(np.uint8))

    # Combine weighted channels
    weighted = (shadow * 0.5 + seasonal * 0.5).astype(np.uint8)
    save_debug_image(weighted, id="01_weighted", output_dir=config['output_dir'])

    # Edge detection
    blur = cv2.GaussianBlur(weighted, (5, 5), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE enhance edges
    sobel = clahe.apply(sobel)

    # Enhance edges in target range
    target = 10
    delta = 10
    strength = 1.0
    
    enhanced = sobel.astype(np.float32)
    diff = enhanced - target
    mask = np.abs(diff) < delta
    enhanced[mask] = target + (diff[mask] * strength)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    save_debug_image(enhanced, id="03_enhanced", output_dir=config['output_dir'])
    
    # Watershed segmentation on enhanced edges
    ret, markers = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    markers = cv2.connectedComponents(markers.astype(np.uint8))[1]
    markers = markers + 1
    markers[enhanced < 30] = 0
    
    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), markers)
    watershed_mask = (markers > 1).astype(np.uint8) * 255
    save_debug_image(watershed_mask, id="03a_watershed", output_dir=config['output_dir'])
        
    return watershed_mask

def remove_small_segments(masks, image_shape, image, min_segment_size, pixel_value_threshold):
    """
    Remove small segments from multiple masks and combine them.

    Args:
        masks (list): List of mask dictionaries from SAM
        image_shape (tuple): Shape of the original image
        image (numpy.ndarray): Original image
        min_segment_size (float): Minimum segment size as fraction of total image
        pixel_value_threshold (float): Threshold for average pixel value
    """
    combined_mask = np.zeros(image_shape, dtype=bool)
    total_pixels = image_shape[0] * image_shape[1]
    
    valid_segments = []
    for mask in masks[1:]:
        segment = mask['segmentation']
        if (np.sum(segment)/total_pixels >= min_segment_size):
            segmented_pixels = image[segment]
            avg_pixel_value = np.mean(segmented_pixels)/255
            num_pixels = np.sum(segment)
            if (avg_pixel_value >= pixel_value_threshold) or 0.8 >= num_pixels/total_pixels >= min_segment_size:
                valid_segments.append(segment)

    kernel = np.ones((5, 5), np.uint8)
    for segment in valid_segments:
        eroded = cv2.erode(segment.astype(np.uint8), kernel, iterations=2)
        combined_mask = np.logical_or(combined_mask, eroded.astype(bool))
    
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
    if np.sum(segment) < 8000:
        return label
    dilated_segment = ndimage.binary_dilation(segment, structure=kernel)
    nodata_count = np.sum(dilated_segment & nodata_mask)
    segment_size = np.sum(dilated_segment)
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
    
    print(f'Number of nodata pixels: {np.sum(nodata_mask)}')
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

def segment_image(image, mask_generator, target_size=(1024, 1024), min_segment_size=0.01, 
                 pixel_value_threshold=0.4, show_masks=False):
    """
    Apply the SAM model to the image and process the resulting masks.

    Args:
        image (numpy.ndarray): Input image
        mask_generator (SamAutomaticMaskGenerator): SAM mask generator
        target_size (tuple): Target size for SAM processing (default: (1024, 1024))
        min_segment_size (float): Minimum segment size as fraction of image (default: 0.01)
        pixel_value_threshold (float): Threshold for avg pixel value (default: 0.4)
        show_masks (bool): Whether to display mask visualization (default: False)
    """
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    
    size = image.shape[:2]
    print(f'Size: {size}')
    
    # Resize image to target size keeping aspect ratio
    image = scale_for_sam(image, target_size)
    
    # Generate masks
    with torch.amp.autocast('cuda'):
        masks = mask_generator.generate(image)
        
    if show_masks:
        show_masks(image, masks, random_colors=True)
        
    # Remove small segments and combine masks
    combined_mask = remove_small_segments(
        masks=masks,
        image_shape=image.shape[:2],
        image=image,
        min_segment_size=min_segment_size,
        pixel_value_threshold=pixel_value_threshold
    )
    
    combined_mask = scale_for_sam(combined_mask, size)
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

def remove_outliers(mask, image, outlier_threshold=1.5):
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

def hremove_outliers(mask, image, outlier_threshold=1.5, size_weight=0.2):
    """
    Detect potential dams and remove them from the mask using multiple metrics.

    Args:
        mask (numpy.ndarray): Input binary mask.
        image (numpy.ndarray): Original image.
        outlier_threshold (float): Z-score threshold for outlier detection.
        size_weight (float): Weight factor for segment size (0-1).

    Returns:
        numpy.ndarray: Updated mask with potential dams removed.
    """
    labeled_mask, num_features = ndimage.label(mask)
    
    # Handle channel-first format
    if len(image.shape) == 3:
        if image.shape[0] in [3, 4]:
            image = image.transpose(1, 2, 0)
            
    if num_features == 0:
        return mask
    
    # Lists to store segment statistics
    segment_stats = []
    segment_sizes = []
    segment_circularity = []
    
    # Calculate total image area for size normalization
    total_area = mask.shape[0] * mask.shape[1]
    
    for i in range(1, num_features + 1):
        segment = (labeled_mask == i)
        segment_pixels = image[segment]
        
        if len(segment_pixels) > 0:
            # Calculate RGB standard deviation
            std_rgb = np.std(segment_pixels[:, :3], axis=0)
            
            # Calculate segment size and normalize
            size = np.sum(segment) / total_area
            
            # Calculate circularity
            contours, _ = cv2.findContours(segment.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contour = contours[0]
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            else:
                circularity = 0
                
            segment_stats.append(std_rgb)
            segment_sizes.append(size)
            segment_circularity.append(circularity)
    
    # Convert to numpy arrays
    segment_stats = np.array(segment_stats)
    segment_sizes = np.array(segment_sizes)
    segment_circularity = np.array(segment_circularity)
    
    # Calculate z-scores for each metric
    rgb_z_scores = stats.zscore(segment_stats, axis=0)
    size_z_scores = stats.zscore(segment_sizes)
    circularity_z_scores = stats.zscore(segment_circularity)
    
    # Apply size-based weighting
    size_factors = 1 + (size_weight * size_z_scores)
    size_factors = np.clip(size_factors, 1 - size_weight, 1 + size_weight)
    
    # Combine metrics with size weighting
    outliers = (
        (np.any(rgb_z_scores < -outlier_threshold * size_factors[:, np.newaxis], axis=1)) |  # RGB variation
        (circularity_z_scores > outlier_threshold * size_factors) |  # Too circular
        (circularity_z_scores < -outlier_threshold * size_factors)   # Too irregular
    )
    import matplotlib.pyplot as plt
    # Plot the segments and their metrics
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original image with segments
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image with Segments')
    
    # Create a colormap for the segments
    colors = plt.cm.rainbow(np.linspace(0, 1, num_features))
    segment_overlay = np.zeros((*mask.shape, 4))
    
    for i in range(num_features):
        segment = labeled_mask == (i + 1)
        color = colors[i]
        if outliers[i]:
            color = [1, 0, 0, 0.3]  # Red for outliers
        else:
            color = [*color[:3], 0.3]  # Original color with transparency
        segment_overlay[segment] = color
    
    plt.imshow(segment_overlay)
    
    # Plot 2: Metrics visualization
    plt.subplot(132)
    plt.scatter(circularity_z_scores, np.mean(rgb_z_scores, axis=1), 
                c=size_z_scores, cmap='viridis')
    plt.colorbar(label='Size Z-Score')
    plt.xlabel('Circularity Z-Score')
    plt.ylabel('Mean RGB Std Z-Score')
    plt.title('Segment Metrics')
    
    # Plot 3: Size distribution
    plt.subplot(133)
    plt.hist(segment_sizes, bins=20)
    plt.xlabel('Normalized Segment Size')
    plt.ylabel('Count')
    plt.title('Segment Size Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Remove outlier segments
    for i, is_outlier in enumerate(outliers):
        if is_outlier:
            mask[labeled_mask == (i + 1)] = False
            
    return mask

def process_single_file(input_file, output_dir, config, mask_generator, preprocess=True):
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
        print(f"Loading existing downscaled image: {downscaled_file}")
        with rasterio.open(downscaled_file) as src:
            downscaled_image = src.read()
            output_profile = src.profile.copy()
    else:
        downscaled_image, output_profile = downscale_tif(input_file, config)
        print("Saving downscaled image...")
        with rasterio.open(downscaled_file, 'w', **output_profile) as dst:
            dst.write(downscaled_image)
        
    if config.get('skip_enabled', False):
        skip = input("Skip segmentation? (y/n): ")
        if skip.lower() == 'y':
            return False, downscaled_image, output_profile
        
    image = downscaled_image
    
    if config.get('autumn_filtered', False):
        print("Applying autumn filter...")
        image = apply_autumn_filter(image)
        
    save_debugs = config.get('save_debug', False)    
    if save_debugs:
        save_debug_image(image, f"{unique_id}", output_dir, "_01_original",)

    config["output_dir"] = output_dir

    # Threshold mask creation
    image = image.transpose(1, 2, 0)
    if preprocess:
        sam_image = create_threshold_mask(image, config)
    else:
        if image.shape[2] == 4:
            sam_image = image[:, :, :3]
    
    
    # Segmentation
    segmentation_mask = segment_image(
        image=sam_image,
        mask_generator=mask_generator,
        target_size=config['segmentation']['sam_target_size'],
        min_segment_size=config['segmentation']['min_segment_size'],
        pixel_value_threshold=config['segmentation']['pixel_value_threshold'],
        show_masks=config.get('show_masks', False)
    )
        
    if save_debugs:
        save_debug_image(segmentation_mask*255, f"{unique_id}", output_dir, "_03_segmentation_mask")
       
    # No-data removal
    nodata_value = np.array(config['segmentation']['nodata_value'])    
    max_nodata_percentage = config['segmentation']['max_nodata_percentage']
    border_size = config['segmentation'].get('border_size', 3)
    
    cleaned_mask = remove_nodata_segments(segmentation_mask, image, nodata_value, max_nodata_percentage, border_size, num_threads=12)
        
    if save_debugs:
        save_debug_image(cleaned_mask*255, f"{unique_id}", output_dir, "_04_no_data_cleaned_mask")
    
    # Dam removal
    outlier_threshold = config['segmentation']['outlier_threshold']
    final_mask = remove_outliers(cleaned_mask, image, outlier_threshold)
    if save_debugs:
        save_debug_image(final_mask*255, f"{unique_id}", output_dir, "_Final")
        
    # check if more than 30% of the area is segmented
    if np.sum(final_mask) /  (255 * final_mask.shape[0] * final_mask.shape[1]) < 0.15:
        print(f"Segmented area is too Small: {np.sum(final_mask)} pixels")
        return False, image, output_profile
    # Final mask processing
    final_mask = keep_largest_segment(np.logical_not(final_mask))
    final_mask = np.where(final_mask == 1, 0, -999)

    # Save final mask
    mask_profile = output_profile.copy()
    mask_profile.update(dtype=rasterio.float32, count=1, nodata=-999)
    with rasterio.open(mask_file, 'w', **mask_profile) as dst:
        dst.write(final_mask.astype(rasterio.float32), 1)

    return True, image, output_profile

def main(config_file, root_dir):
    config = load_config(config_file)
    config['root_dir'] = root_dir
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

    reprocess = []
    for input_file in matching_files:
        success, _, _ = process_single_file(input_file, output_dir, config, mask_generator, preprocess=True)
        if success:
            print(f"Successfully processed {input_file}")
        else:
            reprocess.append(input_file)
            print(f"Failed to process {input_file}")

    if len(reprocess) != 0:
        mask_generator = load_sam_model(config['segmentation']['unprocessed_settings'])
        
    for input_file in reprocess:
        success, _, _ = process_single_file(input_file, output_dir, config, mask_generator, preprocess=False)
        if success:
            print(f"Successfully processed {input_file}")
        else:
            print(f"Failed to process {input_file}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchard Downscaling and Segmentation")
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("root_dir", type=str, help="Root directory to search for files")
    args = parser.parse_args()
    main(args.config, args.root_dir)

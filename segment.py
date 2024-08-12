import rasterio
import numpy as np
from skimage import filters
from skimage.morphology import dilation, binary_dilation, disk, opening, closing, remove_small_objects

def read_file(file_path: str):
    """Read a file from disk."""
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
        nodata = src.nodata
    return data, profile, nodata

def get_ndvi(nir, red):
    """Calculate NDVI from NIR and red bands."""
    return (nir - red) / (nir + red)

def halve_array_size(arr):
    """Reduce the size of the array by half in both dimensions."""
    return arr[::2, ::2]

def segment_image(ndvi: np.ndarray, threshold: float = 0.5, min_size: int = 100):
    """Segment an image using simple thresholding and morphological operations."""
    # Apply threshold
    threshold_value = filters.threshold_otsu(ndvi)
    binary_mask = ndvi > threshold_value
    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)
   
    # Morphological operations
    selem = disk(20)  # Reduced from 50 to 25 due to halved image size
    dilated_mask = dilation(cleaned_mask, disk(30))  # Reduced from 80 to 40
    opened_mask = closing(dilated_mask, selem)
    return opened_mask

def write_tif(output_path: str, data: np.ndarray, profile: dict, original_nodata: float):
    """Write a numpy array to a GeoTIFF file."""
    # Replace NoData values with 0
    data[data == original_nodata] = 0
   
    # Update profile to set nodata value to 0 and adjust resolution
    profile.update(
        dtype=rasterio.uint8, 
        count=1, 
        nodata=0,
        width=data.shape[1],
        height=data.shape[0],
        transform=profile['transform'] * profile['transform'].scale(2, 2)
    )
   
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(rasterio.uint8), 1)

if __name__ == "__main__":
    # Read NIR and Red bands
    nir, profile, nodata_value = read_file("F:/Data/UOG_1676/orthos/data-analysis/nir.tif")
    red, _, _ = read_file("F:/Data/UOG_1676/orthos/data-analysis/red.tif")
   
    # Halve the size of both arrays
    nir = halve_array_size(nir)
    red = halve_array_size(red)
   
    # Calculate NDVI
    ndvi = get_ndvi(nir, red)
    del nir, red
    
    # Segment the image
    mask = segment_image(ndvi, min_size=25)  # Reduced min_size due to halved image
    
    # Write the output mask to a GeoTIFF, replacing NoData with 0
    write_tif("mask.tif", mask, profile, nodata_value)
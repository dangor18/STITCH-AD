import rasterio
from rasterio.enums import Resampling
import os
import tqdm
import numpy as np
from skimage import filters
from skimage.morphology import dilation, binary_dilation, disk, opening, closing, remove_small_objects, area_closing
from skimage.measure import label, regionprops

def read_file(file_path: str):
    """Read a file from disk."""
    with rasterio.open(file_path) as src:
        data = src.read(1)
        profile = src.profile
        nodata = src.nodata
    return data, profile, nodata

def scale_rasters(files, factor=4):
    os.makedirs("temp", exist_ok=True)
    for file_name in tqdm.tqdm(files, desc="Scaling files"):
        # open raster
        with rasterio.open(file_name) as src:
            nodata = src.nodata
                
            color_interps = src.colorinterp

            # upscale to match width and height
            data = src.read(
                out_shape=(
                    src.count,
                    src.height // factor,
                    src.width // factor,
                ),
                resampling=Resampling.bilinear
            )

            # adjust metadata
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )

            output_dir = "temp"
            output_path = os.path.join(output_dir, os.path.basename(file_name))
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=src.height // factor,
                width=src.width // factor,
                count=src.count,
                dtype=data.dtype,
                crs=src.crs,
                transform=transform,
                nodata=nodata
            ) as dst:
                dst.write(data)
                dst.colorinterp = color_interps

def get_pixel_size(file, pixel_size_deg):
    """
    given a patch size in degrees, calculate the number of pixels for the patch size (degrees to number of pixels x and y)
    """
    with rasterio.open(file) as src:
        # get the pixel size in degrees
        pixel_size_x = src.transform.a
        pixel_size_y = src.transform.e

        e = 10 ** -6

        # calculate the number of pixels for the patch size
        pixel_size_x = int(pixel_size_deg / pixel_size_x) * e
        pixel_size_y = int(pixel_size_deg / pixel_size_y) * e

        return int(pixel_size_x)

def get_ndvi(nir, red):
    """Calculate NDVI from NIR and red bands."""
    return (nir - red) / (nir + red)

def segment_image(ndvi: np.ndarray, width, height, min_size: int = 50, border_ratio = 0.12):
    """Segment an image using simple thresholding and morphological operations."""
    # apply threshold
    threshold_value = filters.threshold_otsu(ndvi)
    binary_mask = ndvi > threshold_value
    # Calculate border width and height
    border_width = int(width * border_ratio)
    border_height = int(height * border_ratio)
    
    # Remove border
    binary_mask[:border_height, :] = 0
    binary_mask[-border_height:, :] = 0
    binary_mask[:, :border_width] = 0
    binary_mask[:, -border_width:] = 0

    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size) # remove noise
   
    # Morphological operations
    selem = disk(0.25 * min_size)
    dilated_mask = dilation(cleaned_mask, selem)
    #dilated_mask = remove_small_objects(dilated_mask, min_size=200*min_size)

    # Label connected components
    labeled = label(dilated_mask)
    
    # Get properties of labeled regions
    regions = regionprops(labeled)
    
    # Sort regions by area, descending order
    sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
    
    # Create a mask with the two largest regions
    mask = np.zeros_like(dilated_mask, dtype=bool)
    for region in sorted_regions[:2]:  # Take top 2 regions
        mask[region.coords[:, 0], region.coords[:, 1]] = True
    
    # Apply mask to original image
    #result = image * mask

    #dilated_mask = opening(dilated_mask, selem)
    #opened_mask = area_closing(dilated_mask, area_threshold=min_size, connectivity=1)
    #opened_mask = closing(dilated_mask, disk(min_size))
    return mask

def write_tif(output_path: str, data: np.ndarray, profile: dict):
    """Write a numpy array to a GeoTIFF file."""
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)

def get_dims(file):
    """Get the dimensions of a raster."""
    with rasterio.open(file) as src:
        width = src.width
        height = src.height
    return width, height


if __name__ == "__main__":
    #scale_rasters(["D:/Data/UOG_1676/orthos/data-analysis/nir.tif", "D:/Data/UOG_1676/orthos/data-analysis/red.tif"], factor=4)
    nir, profile, _ = read_file("temp/nir.tif")
    red, _, _ = read_file("temp/red.tif")
    # Calculate NDVI
    ndvi = get_ndvi(nir, red)
    del nir, red
    
    min_size = get_pixel_size("temp/red.tif", 120)
    width, height = get_dims("temp/red.tif")
    #print(min_size)
    print("SEGMENTING")
    mask = segment_image(ndvi, width, height, min_size=min_size)

    write_tif("mask21.tif", mask, profile)
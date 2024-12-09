import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.env import Env

import numpy as np

import tqdm
import tqdm.contrib
import tqdm.contrib.itertools

def getTIFDimensions(file_name):
    """
    RETURNS WIDTH AND HEIGHT FOR TIF FILE
    """
    with rasterio.open(file_name) as file:
        return file.width, file.height

def get_patch_size(temp_dir, patch_size_deg, overlap_size_deg):
    """
    given a patch size in degrees, calculate the number of pixels for the patch size (degrees to number of pixels x and y)
    """
    # all files have been clipped to the same extent and scaled to the same resolution and therefore you can read pixel size from any of the files
    files = os.listdir(temp_dir)
    reference = files[0]

    with rasterio.open(os.path.join(temp_dir, reference)) as src:
        # get the pixel size in degrees
        pixel_size_x = src.transform.a
        pixel_size_y = src.transform.e

        e = 10 ** -6

        # calculate the number of pixels for the patch size
        patch_size_x = int(patch_size_deg / pixel_size_x) * e
        patch_size_y = int(patch_size_deg / pixel_size_y) * e
        overlap_size_x = int(overlap_size_deg / pixel_size_x) * e
        overlap_size_y = int(overlap_size_deg / pixel_size_y) * e

        return int(patch_size_x), abs(int(patch_size_y)), int(overlap_size_x), abs(int(overlap_size_y))
    
def getReferenceGrid(reference_path, scaled_width, scaled_height):
    with rasterio.open(reference_path) as ref:
        # crs and bounds for reference file
        crs = ref.crs
        bounds = ref.bounds
        
        # calc new transform using the ref transform and the scaled width and height from the RGB * scale_factor
        new_transform = ref.transform * ref.transform.scale(
            (ref.width / scaled_width),
            (ref.height / scaled_height)
        )
        
        return {
            'crs': crs,
            'transform': new_transform,
            'width': scaled_width,
            'height': scaled_height,
            'bounds': bounds
        }

def reprojectTIF(files, scaled_width, scaled_height, output_dir, reference, verbose=False):
    """
    CLIPS RASTER EXTENT ACCORDING TO REFERENCE TIF's EXTENT AND RESAMPLES TO A NEW WIDTH AND HEIGHT
    """
    os.makedirs(output_dir, exist_ok=True)
    reference_grid = getReferenceGrid(reference, scaled_width, scaled_height)
    for file_name in tqdm.tqdm(files, desc="Reprojecting Bands", disable=verbose):
        local_name = os.path.basename(file_name)
        output_path = os.path.join(output_dir, local_name)
        if verbose:
            print(os.path.join(output_dir, local_name))

        if not os.path.exists(output_path):
            # open raster
            with Env(GDAL_NUM_THREADS='ALL_CPUS'):
                with rasterio.open(file_name) as src:
                    # read source file metadata
                    src_crs = src.crs
                    src_transform = src.transform
                    src_nodata = src.nodata
                    src_dtype = src.dtypes[0]
                    src_count = src.count
                    
                    # destination metadata
                    dst_meta = src.meta.copy()
                    dst_meta.update({
                        'crs': reference_grid['crs'],
                        'transform': reference_grid['transform'],
                        'width': reference_grid['width'],
                        'height': reference_grid['height']
                    })
                    
                    # Initialize destination array
                    dst_data = np.empty((src_count, reference_grid['height'], reference_grid['width']), dtype=src_dtype)
                    
                    # Perform reprojection for each band
                    for i in range(1, src_count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=dst_data[i - 1],
                            src_transform=src_transform,
                            src_crs=src_crs,
                            dst_transform=reference_grid['transform'],
                            dst_crs=reference_grid['crs'],
                            resampling=Resampling.bilinear,
                            src_nodata=src_nodata,
                            dst_nodata=src_nodata
                        )
                    
                    # Write the reprojected raster to disk
                    with rasterio.open(output_path, 'w', **dst_meta) as dst:
                        dst.write(dst_data)
        else:
            if verbose:
                print(f"{file_name} ALREADY EXISTS IN FOLDER")
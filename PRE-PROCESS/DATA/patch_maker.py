import argparse
import itertools
import numpy as np
import os
import rasterio
from rasterio.windows import Window

from patch_utils import getTIFDimensions, get_patch_size, reprojectTIF

import re
import shutil
import tqdm
import tqdm.contrib
import tqdm.contrib.itertools
import yaml

import multiprocessing
import time

def init_workers(SRC_LIST, mask_path, nodata_ref_path):
    global SRC, MASK, NODATA_REF
    SRC = []
    for src_path, nb in SRC_LIST:
        src_ds = rasterio.open(src_path)
        SRC.append((src_ds, nb))

    MASK = rasterio.open(mask_path)
    NODATA_REF = rasterio.open(nodata_ref_path)

def process_patch(args):
    """
    PROCESS A PATCH FOR A SINGLE FILE
    args: orchard_id, i, j, x, y, block_size_x, block_size_y, anomaly_threshold, output_dir
    """
    (orchard_id, i, j, x, y, block_size_x, block_size_y, anomaly_threshold, output_dir) = args

    # read in block from upscaled image
    win = Window(x, y, block_size_x, block_size_y)
    # read in block from mask
    mask_block = MASK.read(1, window=win)
    nodata_ref_block = NODATA_REF.read(1, window=win)
                    
    # ignore these regions
    if (np.any(mask_block == 0) or np.any(nodata_ref_block == -32767)):
        return

    # stack each band data to block array
    band_data_list = []
    for src, num_bands in SRC:
        # read within window from all the bands
        data = [src.read(b, window=win) for b in range(1, int(num_bands) + 1)]
        band_data_list.extend(data)

    block = np.stack(band_data_list, axis=0)
    
    case_1_count = np.count_nonzero(mask_block == 1)
    case_2_count = np.count_nonzero(mask_block == 2)
    case_3_count = np.count_nonzero(mask_block == 3)
                    
    # check for anomaly cases
    if case_1_count > anomaly_threshold * block_size_x * block_size_y:
        block_file_name = os.path.join(os.path.join(output_dir, "case_1"), f"{orchard_id}_block_{i}_{j}.npy")
    elif case_2_count > anomaly_threshold * block_size_x * block_size_y:
        block_file_name = os.path.join(os.path.join(output_dir, "case_2"), f"{orchard_id}_block_{i}_{j}.npy")
    elif case_3_count > anomaly_threshold * block_size_x * block_size_y:
        block_file_name = os.path.join(os.path.join(output_dir, "case_3"), f"{orchard_id}_block_{i}_{j}.npy")
    elif case_1_count + case_2_count + case_3_count == 0:
        block_file_name = os.path.join(os.path.join(output_dir, "normal"), f"{orchard_id}_block_{i}_{j}.npy")
    else:
        return
                            
    np.save(block_file_name, block.transpose(1, 2, 0))

def create_patches(orchard_id, files, dimensions, scaled_width, scaled_height, block_size, overlap, anomaly_threshold, output_dir, mask_file="mask.tif", reference="reg.tif",  verbose=False):
    """
    CREATES BLOCK BIN FILES FOR VARIOUS DIMENSIONS
    PARAMETERS: Root path to files(for block naming), List of file names, list of bands for each file, desired width and height to upscale to, block size, block overlap, output directory for chunks
    """
    SRC_LIST = []
    # make directories
    output_subdirs = ["case_1", "case_2", "case_3", "normal"]
    for subdir in output_subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # reproject all bands
    temp_dir = os.path.join(output_dir, "temp/")
    # include the reference TIF and the mask file in the list of files to reproject
    files.append(mask_file), files.append(reference)
    files = list(map(lambda f: f.replace('\\', '/'), files))
    reprojectTIF(files, scaled_width, scaled_height, temp_dir, reference=reference, verbose=verbose)

    block_size_x, block_size_y, overlap_x, overlap_y = get_patch_size(temp_dir, block_size, overlap)
    print(f"Block size: {block_size_x}x{block_size_y} px, Overlap: {overlap_x}x{overlap_y} px")

    mask_path = os.path.join(temp_dir, os.path.basename(mask_file))
    nodata_ref_path = os.path.join(temp_dir, os.path.basename(reference))

    for file_name, num_bands in zip(files, dimensions):
        local_name = file_name.split("/")[-1]
        src_path = os.path.join(temp_dir, local_name)
        SRC_LIST.append((src_path, num_bands))
        
    # block starting positions
    y_starts = np.arange(0, scaled_height - block_size_y + 1, block_size_y - overlap_y)
    x_starts = np.arange(0, scaled_width - block_size_x + 1, block_size_x - overlap_x)

    loop_product = itertools.product(enumerate(y_starts), enumerate(x_starts))
    patch_args = []
    # extract patches for a channel in parallel
    for (i, y), (j, x) in loop_product:
        patch_args.append((
            orchard_id,
            i,
            j,
            x,
            y,
            block_size_x,
            block_size_y,
            anomaly_threshold,
            output_dir
        ))

    with multiprocessing.Pool(processes=os.cpu_count(),
                                initializer=init_workers, 
                                initargs=(SRC_LIST, mask_path, nodata_ref_path)) as pool:
        pool.map(process_patch, patch_args)
    
    # delete temp upscale tif folder and contents
    shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="A tool for chunking large orthomosaic TIF files into smaller patches.")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    parser.add_argument("config", type=str, help="path to the .yaml configuration file")

    args = parser.parse_args()
    cwd = os.path.dirname(os.path.realpath(__file__))       # directory of the script

    config_path = os.path.join(cwd, args.config)
    verbose = args.verbose

    # try open config file
    with open(config_path, "r") as f:
        config_documents = yaml.safe_load_all(f)
        for _, config in enumerate(config_documents):
            # PARAMETERS
            try:
                path = config["path"]
                files = []
                dimensions = []
                for file in config["files"]:
                    files.append(os.path.join(path, file["name"]))
                    dimensions.append(file["dimensions"])

                chunk_size = config["chunk_size"]
                chunk_overlap = config["chunk_overlap"]
                anomaly_threshold = config["anomaly_threshold"]
                scale_ratio = config["scale_ratio"]
                    
                # get width and height of RGB file for orchard and times each by scale factor (new height and width of each channel for this particular orchard)
                rgb_path = os.path.join(path, "orthos", "export-data", "orthomosaic_visible.tif")
                width, height = getTIFDimensions(rgb_path)

                scale_width = int(scale_ratio * width)
                scale_height = int(scale_ratio * height)

                if verbose:
                    print(f"Scaling every channel to: {scale_width}x{scale_height}")
                    
                mask_file = config["mask"]
                # reference file used to determine nodata regions and clip the rasters against before upscaling
                reference_file = config["reference"]
                output_dir = os.path.join(cwd, config["output_path"])

            except KeyError as e:
                field = re.findall(r"'(.+?)'", str(e))[-1]
                print(f"Missing required field '{field}' in .yaml configuration document.")
                exit()
                
            print(f"Chunking project: {path}")
            orchard_id = os.path.basename(path)
            create_patches(orchard_id, files, dimensions, scale_width, scale_height, chunk_size, chunk_overlap, anomaly_threshold, output_dir, mask_file, reference_file, verbose=verbose)

if __name__ == "__main__":
    try:
        start = time.time()
        main()
        print(f"Execution time: {time.time() - start:.2f} seconds")
    except KeyboardInterrupt:
        print("Cancelling job...")
        exit()
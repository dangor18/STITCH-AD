# STITCH-AD
STITCH-AD is a novel pipeline for the accurate detection of stitching artefacts in orthomosaic imagery caused by registration errors. It combines deep learning models with classical algorithms to detect and isolate possible artefacts.

# Disclaimer
The data used to train models and use for inference will not be publicly available. Example orthomosaics can be found at: https://dronemapper.com/sample_data/

### Requirements
Use python 3.9.5

## Code structure
There are 4 major sections to the pipeline: **data preparation, segmentation, model training, and orchard (orthomosaic) level classification**.
The coded is structured as follows:

📁 Project Root
+ 📁 ORCHARD_AD
  + 📁 configs
  + 📄 inference_utils.py

+ 📁 PATCH_AD  
  + 📁 RD
  + 📁 UniAD

+ 📁 PRE_PROCESS
  + 📁 DATA
  + 📁 SEGMENTATION

+ 📄 inference.py (your entry point to the orchard classification)

Reverse Distillation (RD) and UniAD were the two models evaluated for patch level anomaly detection. The DATA folder contains relevant code for turning orthomosaics into patches saved as .npy files.

### Execution
patch.bat illustrates an example usage of the scripts used to create patches, create a train test split, and a corresponding meta file. Run anyone of the models, specifying the patch directory and meta file path.

Below is an example config for patch extraction:
### Example:
```yaml
path: A:/STITCH-O/Ortho-1
output_path: patches/1/
files:
  -
    name: orthos/data-analysis/lwir.tif
    dimensions: 1
  -
    name: orthos/data-analysis/red.tif
    dimensions: 1
  -
    name: orthos/data-analysis/reg.tif
    dimensions: 1
  -
    name: orthos/raster.tif
    dimensions: 1
  -
    name: orthos/export-data/orthomosaic_visible.tif
    dimensions: 3
mask: A:/Uploads/mask_1.tif
reference: A:/STITCH-AD/ortho-1/orthos/data-analysis/reg.tif
chunk_size: 256
chunk_overlap: 50
anomaly_threshold: 0.7
scale_ratio: 0.5
---
repeat for next orchard
```
```
Data Organisation:

A:/STITCH-O/Ortho-1
├── orthos
    ├── data-analysis
    │   ├── lwir.tif
    │   ├── red.tif
    │   └── reg.tif
    ├── raster.tif
    └── export-data
        └── orthomosaic_visible.tif

Output Example:

working directory
├── chunker.py
└── chunks
    ├──  1
    |   ├── test
    |   |   ├── Case_1
    |   |   |   └── ortho1_block_x_y.npy
    |   |   |   └── ortho1_block_m_n.npy
    |   |   |   └── ...
    |   |   ├── Case_2
    |   |   |   └── ...
    |   |   └── Normal
    |   |       └── ...
    |   └── train
    |       └── Normal
    |           └── ...   
    ├── 2
    |   ├── test
    |   |   ├── Case_1
    |   |   |   └── ortho2_block_x_y.npy
    |   |   |   └── ortho2_block_m_n.npy
    |   |   |   └── ...
    |   |   ├── Case_2
    |   |   |   └── ...
    |   |   └── Normal
    |   |       └── ...
    |   └── train
    |       └── Normal
    |           └── ...
    └── metadata
        ├── test_metadata.json
        └── train_metadata.json
```

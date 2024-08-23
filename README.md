## Setting up the environment
pip install -r requirements.txt
## Patch Making Pipeline:
Linux:
```shell
> pipeline.sh
```
Windows:
```shell
> pipeline.bat
```
This will run three files in sequence:
  1. `chunker.py`
  2. `process_chunks.py`
  3. `generate_metadata.py`

And assumes a correctly formatted config file with the default name `config.yaml`.
Config files can be added to the script given to chunker.py

## Chunk / Patch Creator

```shell
> python chunker.py [config file]
```

### Config Parameters:

- `path`: The path to the directory containing the images to be chunked. The path can be either absolute or relative to the current working directory.
- `output_path`: The path to the directory where the chunked images will be saved. The path can be either absolute or relative to the current working directory.
- `files`: A list of files to be used as layers in the dataset. All files must be located in the given `path` directory.
- `name`: The path to the file.
- `dimensions`: The number of dimensions of the file. The number of dimensions is used to determine the number of channels in the image.
- `mask`: The path to the .tif file to be used a a mask. The mask file must be located in the path directory.
- `reference`: The path to the .tif file to be used for clipping against and for reading in nodata values to skip over for a patch. This doesn't have to be a layer considered for a chunk. DO NOT MAKE RGB.
- `chunk_size`: The width/height of the chunks in 10^-6 of a degree. The chunks are square.
- `chunk_overlap`: The length of the overlap between chunks in 10^-6 of a degree.
- `threshold`: The percentage a chunk must overlap with an anomalous region to be considered an anomalous chunk.
- `scale_ratio`: The factor the width and height the rgb image's dimensions are multiplied by to achieve a resolution all channels for a given orchard are upscaled to.

### Example:
```yaml
path: A:/STITCH-O/UOG_1676
output_path: chunks/
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
mask: A:/Uploads/mask_1676.tif
reference: A:/STITCH-O/UOG_1676/orthos/data-analysis/reg.tif
chunk_size: 256
chunk_overlap: 50
anomaly_threshold: 0.7
normal_threshold: 0.0
scale_ratio: 0.5
---
repeat for next orchard
```
```
Data Organisation:

A:/STITCH-O/UOG_1676
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
    ├── 1676
    |   ├── test
    |   |   ├── Case_1
    |   |   |   └── UOG_1676_block_x_y.npy
    |   |   |   └── UOG_1676_block_m_n.npy
    |   |   |   └── ...
    |   |   ├── Case_2
    |   |   |   └── ...
    |   |   └── Normal
    |   |       └── ...
    |   └── train
    |       └── Normal
    |           └── ...   
    ├── 1996
    |   ├── test
    |   |   ├── Case_1
    |   |   |   └── UOG_1996_block_x_y.npy
    |   |   |   └── UOG_1996_block_m_n.npy
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

## Running the patch level model
- To train the regular Reverse Distillation model from scratch, run RD.py --config [config_file]
- To train the contrastive learning RD model, run RD_contrast.py --config [config_file]
- To test a saved model and get final results, run RD.py --test --config [config_file]
- To tune a model's parameters with Optuna, adjust which parameters to adjust in the objective function, and run RD.py --tune --config [config_file]

### Config file
- `data_path`: The path to the patch data.
- `meta_path`: The path for the metadata json files.
- `log_path`:  Filename for a file you wish to save logs to (each evaluation during training and average scores for each case)
- `resize_x`: The width to which images will be resized to.
- `resize_y`: The height to which images will be resized to.
- `channels`: The number of channels in the patches (supports 1 or 3 currently).
- `model_path`: The path for saving the best model during evaluation.
- `architecture`: The architecture of the neural network model (Choice: resnet18, resnet50, wide_resnet50_2).
- `bn_attention`: A flag for whether attention is applied to the bottleneck embedding. False or CBAM.
- `loss_weights`: A list of weights for each scale of the loss function.
- `score_weight`: The weight assigned to taking the average of the anomaly to the final score (found to increase results very slightly).
- `proj_loss_weight`: The weight assigned to the projection losses contribution to total model loss. Only used for contrast model.
- `ssot_weight`: The weight assigned to the SSOT (Self-Supervised Optimal Transport) loss component. Only used for contrast model.
- `contrast_weight`: The weight assigned to the contrastive loss component. Only used for contrast model.
- `reconstruct_weight`: The weight assigned to the reconstruction loss component. Only used for contrast model.
- `num_epochs`
- `batch_size`
- `proj_lr`: The learning rate for the projection optimizer. Only used for contrast model. Use `lr` for regular RD.
- `distill_lr`: The learning rate for the distillation optimizer. Only used for contrast model. Use `lr` for regular RD.
- `proj_lr_factor`: The factor by which the projection learning rate is adjusted after every 10 epochs. Only used for contrast model. Use `lr_factor` for regular RD.
- `distill_lr_factor`: The factor by which the distillation learning rate is adjusted after every 10 epochs. Only used for contrast model. Use `lr_factor` for regular RD.
- `beta1_proj`: Beta1 param for projective layer optimizer. Only used for contrast model. Use `beta1` for regular RD.
- `beta2_proj`: Beta2 param for projective layer optimizer. Only used for contrast model. Use `beta2` for regular RD.
- `beta1_distill`: Beta1 param for student and bn optimizer. Use `beta1` for regular RD.
- `beta2_distill`: Beta2 param for student and bn optimizer. Use `beta2` for regular RD.

### Example config for RD_contrast which produced 99.1 average AUROC:
- `data_path`: data/chunks/DRGB_80/
- `meta_path`: data/chunks/DRGB_80/metadata/
- `log_path`: logs/logs.txt
- `resize_x`: 256
- `resize_y`: 256
- `channels`: 3
- `model_path`: checkpoints/RD.pth
- `architecture`: "wide_resnet50_2"
- `bn_attention`: CBAM
- `loss_weights`: [0.5, 0.9, 1.2]
- `score_weight`: 0.4
- `proj_loss_weight`: 0.758175413875071
- `ssot_weight`: 0.015096731092610782
- `contrast_weight`: 0.9872387127026788
- `reconstruct_weight`: 0.6513426603306253
- `num_epochs`: 30
- `batch_size`: 32
- `proj_lr`: 0.001
- `distill_lr`: 0.005
- `proj_lr_factor`: 0.25
- `distill_lr_factor`: 0.25
- `beta1_proj`: 0.5
- `beta2_proj`: 0.999
- `beta1_distill`: 0.5
- `beta2_distill`: 0.999

## Running the orchard level models
- python inference.py --config [config_file]

### Config file
- `data_path`: ...
- `meta_path`: ...
- `resize_x`: ...
- `resize_y`: ...
- `channels`: ...
- `model_type`: Which architecture you're loading. Either RD or RDProj
- `model_path`: ...
- `architecture`: ...
- `bn_attention`: ...
- `score_weight`: ...
- `feature_weights`: ...
- `contamination`: Parameter for ISO forest. Default to 0.05
- `forest_threshold`: Parameter for ISO forest thresholding number of outliers to classify orchard. Set to 25
- `min_cluster_size`: Parameter for HDBSCAN for minimum size of a cluster to be considered as possibly anomalous. Set to 5.
- `v_thresh`: Parameter for HDBSCAN for minimum average score (or height) of cluster above the normalized average of the normal / largest cluster. Set to 1.5.
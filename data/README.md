# Pipeline:
## Usage:

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
  2. `calc_stats.py`
  3. `train_test_split.py`
  4. `baseline_model.py`

And assumes a correctly formatted config file with the name `config.yaml`

# Chunker
## Usage

```shell
> python chunker.py [config file]
```

## YAML Config File

The project uses YAML config files to set parameters for chunking images.

### Parameters:

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

### Example 1:
```
working directory
├── chunker.py
└── chunks
    └── Case_1
        └── UOG_1676_block_x_y.npy
        └── UOG_1996_block_m_n.npy
        └── ...
    └── Case_2
        └── ...
    └── Normal
        └── ...

A:/STITCH-O/UOG_1676
├── orthos
    ├── data-analysis
    │   ├── lwir.tif
    │   ├── red.tif
    │   └── reg.tif
    ├── raster.tif
    └── export-data
        └── orthomosaic_visible.tif
```

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

# Show.py / Visualizer
## Usage:
  python show.py --chunk_dir chunks/ --orchard_id 1676

  The above will show chunks for a specific orchard.
  
  or
  
  python show.py

  The above will show all chunks

# Baseline Model
## Baseline model config example
Note: Higher batch size = more memory usage
At a batch size of 8, the model uses 6GB of RAM and 4.8gb of VRAM
At a batch size of 16, the model uses 7GB of RAM and 8gb of VRAM

```yaml
num_epochs: 10
batch_size: 8
learning_rate: 0.001
weight_decay: 0.0001
data_path: chunks/
data_stats_path: data_stats/
model_path: model.pth
```

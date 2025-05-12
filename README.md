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

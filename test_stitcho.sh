#!/bin/bash
#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:ampere:1
#SBATCH --time=1:00:00
#SBATCH --job-name="STITCH-O Test"
#SBATCH --mail-user=grddan017@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

CUDA_VISIBLE_DEVICES=$(ncvd)

module load python/miniconda3-py3.9
source activate STITCHO395

python main.py --config configs/model_config_hpc_test.yaml

conda deactivate
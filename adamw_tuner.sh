#!/bin/bash
#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --nodes=1 --ntasks=2 --gres=gpu:ampere:1
#SBATCH --time=48:00:00
#SBATCH --job-name="RD4AD_TUNE_ADAMW"
#SBATCH --mail-user=grddan017@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

CUDA_VISIBLE_DEVICES=$(ncvd)

module load python/miniconda3-py39
pip install --user -r hpc_req.txt
pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118

python main.py --config configs/model_config_adamw.yaml --tune
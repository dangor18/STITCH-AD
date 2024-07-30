@echo off
python data\chunker.py configs\chunk_dem_config_80.yaml
python data\process_chunks.py data\chunks\D_80\ data\chunks\D_80\
python data\generate_metadata.py data\chunks\D_80
pause
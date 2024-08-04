@echo off
python data\chunker.py configs\chunk_config_80.yaml
python data\process_chunks.py data\chunks\DRGB_80\ data\chunks\DRGB_80\
python data\generate_metadata.py data\chunks\DRGB_80
pause
@echo off
python data\chunker.py configs\chunk_config.yaml
python data\process_chunks.py data\chunks\ data\chunks\
python data\generate_metadata.py
pause
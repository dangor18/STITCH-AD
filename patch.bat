@echo off
python PRE-PROCESS/DATA/patch_maker.py configs/patch_config_example.yaml
python PRE-PROCESS/DATA/process_chunks.py PATCHES/DRGB_80/
python PRE-PROCESS/DATA/generate_metadata.py PATCHES/DRGB_80/
pause
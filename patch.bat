@echo off
python PRE-PROCESS/DATA/patch_maker.py configs/patch_config_example.yaml
python PRE-PROCESS/DATA/process_chunks.py STITCH-O_PATCHES/
python PRE-PROCESS/DATA/generate_metadata_expanded.py STITCH-O_PATCHES/
pause
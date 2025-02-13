@echo off
python PRE_PROCESS/DATA/patch_maker.py configs/patch_config.yaml
python PRE_PROCESS/DATA/process_chunks.py STITCH-O_PATCHES/
python PRE_PROCESS/DATA/generate_metadata_expanded.py STITCH-O_PATCHES/
pause
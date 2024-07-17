@echo off
python data\chunker.py configs\chunk_dem_config.yaml
python data\calc_stats.py data\dem_chunks --separate 1
python data\train_test_split.py data\dem_chunks\
python data\scale_data_multi.py
python data\generate_metadata.py
pause
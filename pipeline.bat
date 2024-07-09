@echo off
python data\chunker.py chunk_config.yaml
python data\calc_stats.py data\chunks --separate 3
python data\train_test_split.py data\chunks\
python data\scale_data_multi.py
pause
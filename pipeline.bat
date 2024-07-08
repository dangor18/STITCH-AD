@echo off
python data\chunker.py chunk_config.yaml
python data\calc_stats.py data\chunks\ 3
python data\train_test_split.py data\chunks\
pause
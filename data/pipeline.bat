@echo off
python chunker.py config.yaml
python calc_stats.py chunks/ 4
python train_test_split.py chunks/
python baseline_model.py
pause
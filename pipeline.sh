#!/bin/bash

python chunker.py config.yaml
python calc_stats.py chunks/ 7
python train_test_split.py chunks/
@echo off
python data\chunker.py configs\chunk_config_80.yaml
python data\process_chunks.py data\chunks\DRGB_80\ data\chunks\DRGB_80\
python data\generate_metadata.py data\chunks\DRGB_80

python data\chunker.py configs\chunk_inference_test.yaml
python data\process_chunks.py data\chunks\DRGB_INF_TEST\ data\chunks\DRGB_INF_TEST\
python data\generate_metadata.py data\chunks\DRGB_INF_TEST

python data\chunker.py configs\chunk_inference_train.yaml
python data\process_chunks.py data\chunks\DRGB_INF_TRAIN\ data\chunks\DRGB_INF_TRAIN\
python data\generate_metadata.py data\chunks\DRGB_INF_TRAIN
pause
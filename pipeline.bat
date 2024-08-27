@echo off
python data\chunker.py configs\chunk_test.yaml
python data\process_chunks.py data\chunks\test\
python data\generate_metadata.py data\chunks\test
pause
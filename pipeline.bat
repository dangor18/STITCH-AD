@echo off
python data\chunker.py configs\chunk_demo.yaml
python data\process_chunks.py data\chunks\demo\
python data\generate_metadata.py data\chunks\demo
pause
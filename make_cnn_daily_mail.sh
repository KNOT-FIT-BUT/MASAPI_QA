#!/bin/bash
./run.py huggingface_create_db experiments/cnn_dailymail cnn_dailymail train id article BM25 -n 3.0.0 --threads 6
#./run.py huggingface_create_db experiments/dpr_cnn_dailymail cnn_dailymail test id article DPR -n 3.0.0 --batch 8
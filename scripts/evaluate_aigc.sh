#!/bin/bash

cd /projects/bdec/adhakal2/SimLBR

python -m simlbr.evaluate \
    --devices 1 \
    --dataset_name aigc \
    --data_dir ../data/fake_data/AIGC/AIGCDetectionBenchMark \
    --ckpt_path "$1" \
    --batch_size 200 \
    --num_workers 20

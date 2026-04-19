#!/bin/bash

cd ../

python -m simlbr.train \
    --max_epochs 5 \
    --devices 4 \
    --precision 16-mixed \
    --accelerator gpu \
    --val_check_interval 1.0 \
    --dataset_name aigc \
    --data_dir /projects/bdec/adhakal2/data/fake_data/AIGC/AIGCDetectionBenchMark \
    --train_model ProGAN \
    --val_model combined \
    --ds_fraction 1.0 \
    --batch_size 200 \
    --num_workers 28 \
    --backbone dinov3 \
    --lr 1e-4 \
    --wt_decay 1e-2 \
    --activation relu \
    --hidden_layers 2 \
    --dropout 0.3 \
    --lbr \
    --lbrdist 0.5 0.8 \
    --log_dir /projects/bdec/adhakal2/SimLBR/logs \
    --run_name aigc_simlbr_test \
    --project_name SimLBR_Release 
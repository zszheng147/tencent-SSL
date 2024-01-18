#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

# -m debugpy --listen 55555 --wait-for-client
python \
    -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=33352 --use_env \
    main.py \
    common.mode='train' \
    common.batch_size=256 \
    common.max_epoch=100 \
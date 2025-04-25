#!/bin/bash
export OMP_NUM_THREADS=4

NUM_GPUS=8
MODEL="mae_vit_tiny_patch4"
DATA_PATH="./datasets/cifar10_dataset"
OUTPUT_DIR="./ckpts/mae_bootstrap/pretrain_final"
BATCH_SIZE=128
ACCUM=1
EPOCHS=200
WARMUP_EPOCHS=10
BASE_LR=1.5e-4
INPUT_SIZE=32
MASK_RATIO=0.75
BS_ITERS=5
EMA_DECAY=0.99
EMA_LR_DECAY=0.1
FEATURE_LAYERS=1,6,12

torchrun --nproc_per_node=${NUM_GPUS} --master_port=12362 main_bootstrapped_pretrain.py \
    --world_size ${NUM_GPUS} \
    --model ${MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --accum_iter ${ACCUM} \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --blr ${BASE_LR} \
    --input_size ${INPUT_SIZE} \
    --mask_ratio ${MASK_RATIO} \
    --norm_pix_loss \
    --log_dir "./logs/tb" \
    --bootstrapping \
    --bootstrap_iterations ${BS_ITERS} \
    --ema_decay ${EMA_DECAY} \
    --ema_lr_decay ${EMA_LR_DECAY} \
    --feature_layers ${FEATURE_LAYERS}


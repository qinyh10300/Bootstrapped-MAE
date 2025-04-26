#!/bin/bash
export OMP_NUM_THREADS=4

NUM_GPUS=8
MODEL="mae_vit_tiny_patch4"
DATA_PATH="./datasets/cifar10_dataset"
OUTPUT_DIR="./ckpts/mae_baseline/pretrain_final"
BATCH_SIZE=128
ACCUM=1
EPOCHS=200
WARMUP_EPOCHS=10
BASE_LR=1.5e-4
INPUT_SIZE=32
MASK_RATIO=0.75

torchrun --nproc_per_node=${NUM_GPUS} main_pretrain.py \
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
    --log_dir "./logs/tb"

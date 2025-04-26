#!/bin/bash
export OMP_NUM_THREADS=4

NAME="mae_deit_pretrain"
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="./logs/${NAME}/tb_${NAME}_${CURRENT_DATETIME}"
MODEL="mae_deit_tiny_patch4"
DATA_PATH="./dataset/cifar10_dataset"
OUTPUT_DIR="./ckpts/${NAME}/${CURRENT_DATETIME}"
BATCH_SIZE=256
ACCUM=2
EPOCHS=200
WARMUP_EPOCHS=10
BASE_LR=1.5e-4
INPUT_SIZE=32
MASK_RATIO=0.75

python main_pretrain.py \
    --name ${NAME} \
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
    --log_dir ${LOG_DIR} \
    --device cuda:2 \
    --current_datetime ${CURRENT_DATETIME} \
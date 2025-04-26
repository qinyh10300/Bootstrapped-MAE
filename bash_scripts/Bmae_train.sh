#!/bin/bash
export OMP_NUM_THREADS=4

NAME="Bmae_deit_pretrain"
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="./logs/${NAME}/tb_${CURRENT_DATETIME}"
MODEL="mae_deit_tiny_patch4"
DATA_PATH="./dataset/cifar10_dataset"
OUTPUT_DIR="./ckpts/${NAME}/${CURRENT_DATETIME}"
BATCH_SIZE=256  # modfiy to fit your GPU memory
ACCUM=2
EPOCHS=200
WARMUP_EPOCHS=10
BASE_LR=1.5e-4
INPUT_SIZE=32
MASK_RATIO=0.75
BOOTSTRAP_STEPS=5
BOOTSTRAP_METHOD='last_layer'
EMA_DECAY=0.99

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
    --log_dir ${LOG_DIR}\
    --is_bootstrapping \
    --bootstrap_steps ${BOOTSTRAP_STEPS} \
    --bootstrap_method ${BOOTSTRAP_METHOD} \
    --use_ema \
    --ema_decay ${EMA_DECAY} \
    --device cuda:2 \
    --current_datetime ${CURRENT_DATETIME} \

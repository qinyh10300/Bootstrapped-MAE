#!/bin/bash
export OMP_NUM_THREADS=4

NAME="Bmae_train_deit"
MODEL="mae_deit_tiny_patch4"
DATA_PATH="./dataset/cifar10_dataset"
OUTPUT_DIR="./ckpts/${NAME}/pretrained"
BATCH_SIZE=128  # modfiy to fit your GPU memory
ACCUM=2
EPOCHS=200
WARMUP_EPOCHS=10
BASE_LR=1.5e-4
INPUT_SIZE=32
MASK_RATIO=0.75
BOOTSTRAP_STEPS=5
BOOTSTRAP_METHOD='last_layer'
EMA_DECAY=0.99

# 获取当前日期和时间
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 动态生成日志目录
LOG_DIR="./logs/${NAME}/tb_${NAME}_${CURRENT_DATETIME}"

python main_pretrain.py \
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

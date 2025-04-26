#!/bin/bash
export OMP_NUM_THREADS=4

MODEL="vit_tiny_patch4"
DATA_PATH="./dataset/cifar10_dataset"
OUTPUT_DIR="./ckpts/mae_bootstrap/finetune_final"
BATCH_SIZE=64
EPOCHS=100
WARMUP_EPOCHS=10
BASE_LR=1e-3
INPUT_SIZE=32
WEIGHT_DECAY=0
DROP_PATH=0.05

# 获取当前日期和时间
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 动态生成日志目录
LOG_DIR="./logs/${NAME}/tb_${NAME}_${CURRENT_DATETIME}"

python main_finetune.py \
    --model ${MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --blr ${BASE_LR} \
    --input_size ${INPUT_SIZE} \
    --weight_decay ${WEIGHT_DECAY} \
    --drop_path ${DROP_PATH} \
    --log_dir "./logs/tb" \
    --finetune "./ckpts/mae_bootstrap/pretrain_final/checkpoint-39.pth" \
    --nb_classes 10 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval

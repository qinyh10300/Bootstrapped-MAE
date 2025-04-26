#!/bin/bash
export OMP_NUM_THREADS=4

NAME="Bmae_deit_linear"
MODEL="deit_tiny_patch4"
DATA_PATH="./dataset/cifar10_dataset"
OUTPUT_DIR="./ckpts/Bmae/linprobe"
BATCH_SIZE=256  # modfiy to fit your GPU memory
EPOCHS=100
WARMUP_EPOCHS=10
BASE_LR=0.01
WEIGHT_DECAY=0
INPUT_SIZE=32
CKPT="ckpts/Bmae_train_deit/pretrained/Bmae-5_EMA-39.pth"

# 获取当前日期和时间
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 动态生成日志目录
LOG_DIR="./logs/${NAME}/tb_${NAME}_${CURRENT_DATETIME}"

python main_linprobe.py \
    --model ${MODEL} \
    --input_size ${INPUT_SIZE} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --blr ${BASE_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --log_dir ${LOG_DIR} \
    --finetune ${CKPT} \
    --nb_classes 10 \
    --device cuda:1 \
    --current_datetime ${CURRENT_DATETIME} \
    # --dist_eval  # 是否采用分布式评估

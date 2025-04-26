#!/bin/bash
export OMP_NUM_THREADS=4

NAME="mae_deit_linear"
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="./logs/${NAME}/tb_${NAME}_${CURRENT_DATETIME}"
MODEL="deit_tiny_patch4"
DATA_PATH="./dataset/cifar10_dataset"
OUTPUT_DIR="./ckpts/${NAME}/${CURRENT_DATETIME}"
BATCH_SIZE=256  # modfiy to fit your GPU memory
EPOCHS=100
WARMUP_EPOCHS=10
BASE_LR=0.01
WEIGHT_DECAY=0
INPUT_SIZE=32
CKPT="ckpts/mae_2025-04-26_16-56-35-199.pth"

python main_linprobe.py \
    --name ${NAME} \
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
    --device cuda:0 \
    --current_datetime ${CURRENT_DATETIME} \
    # --dist_eval  # 是否采用分布式评估

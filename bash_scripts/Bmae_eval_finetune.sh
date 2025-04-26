#!/bin/bash
export OMP_NUM_THREADS=4

NAME="Bmae_deit_finetune"
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="./logs/${NAME}/tb_${NAME}_${CURRENT_DATETIME}"
MODEL="deit_tiny_patch4"
DATA_PATH="./dataset/cifar10_dataset"
OUTPUT_DIR="./ckpts/${NAME}/${CURRENT_DATETIME}"
BATCH_SIZE=256
EPOCHS=100
WARMUP_EPOCHS=10
BASE_LR=1e-3
INPUT_SIZE=32
WEIGHT_DECAY=0
DROP_PATH=0.05
CKPT="ckpts/Bmae_train_deit/pretrained/Bmae-5_EMA-39.pth"

python main_finetune.py \
    --name ${NAME} \
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
    --log_dir ${LOG_DIR} \
    --finetune ${CKPT} \
    --nb_classes 10 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --current_datetime ${CURRENT_DATETIME} \
    --device cuda:0 
    # 由于Mixup库源码的问题，这里只能使用cuda:0
    # --dist_eval

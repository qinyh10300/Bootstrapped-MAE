#!/bin/bash
export OMP_NUM_THREADS=4

# 默认值设置
NAME="mae_deit_linear"
MODEL="deit_tiny_patch4"
DATA_PATH="./dataset/cifar10_dataset"
BATCH_SIZE=256  # 修改以适配 GPU 内存
EPOCHS=100
WARMUP_EPOCHS=10
BASE_LR=0.01
WEIGHT_DECAY=0
INPUT_SIZE=32
CKPT="ckpts/mae_2025-04-26_16-56-35-199.pth"
DEVICE="cuda:0"
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --name)
      NAME="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --data_path)
      DATA_PATH="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --warmup_epochs)
      WARMUP_EPOCHS="$2"
      shift 2
      ;;
    --blr)
      BASE_LR="$2"
      shift 2
      ;;
    --weight_decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --input_size)
      INPUT_SIZE="$2"
      shift 2
      ;;
    --finetune)
      CKPT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --current_datetime)
      CURRENT_DATETIME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# 动态生成日志和输出目录
LOG_DIR="./logs/${NAME}/tb_${CURRENT_DATETIME}"
OUTPUT_DIR="./ckpts/${NAME}/${CURRENT_DATETIME}"

# 创建日志目录
mkdir -p ${LOG_DIR}

# 将参数写入 YAML 格式的参数表文件
PARAMS_FILE="${LOG_DIR}/params.yaml"
echo "name: ${NAME}" > ${PARAMS_FILE}
echo "model: ${MODEL}" >> ${PARAMS_FILE}
echo "data_path: ${DATA_PATH}" >> ${PARAMS_FILE}
echo "batch_size: ${BATCH_SIZE}" >> ${PARAMS_FILE}
echo "epochs: ${EPOCHS}" >> ${PARAMS_FILE}
echo "warmup_epochs: ${WARMUP_EPOCHS}" >> ${PARAMS_FILE}
echo "base_lr: ${BASE_LR}" >> ${PARAMS_FILE}
echo "weight_decay: ${WEIGHT_DECAY}" >> ${PARAMS_FILE}
echo "input_size: ${INPUT_SIZE}" >> ${PARAMS_FILE}
echo "finetune: ${CKPT}" >> ${PARAMS_FILE}
echo "device: ${DEVICE}" >> ${PARAMS_FILE}
echo "log_dir: ${LOG_DIR}" >> ${PARAMS_FILE}
echo "output_dir: ${OUTPUT_DIR}" >> ${PARAMS_FILE}
echo "current_datetime: ${CURRENT_DATETIME}" >> ${PARAMS_FILE}

# 执行 Python 脚本
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
    --device ${DEVICE} \
    --current_datetime ${CURRENT_DATETIME}
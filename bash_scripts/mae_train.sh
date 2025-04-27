#!/bin/bash
export OMP_NUM_THREADS=4

# 默认值设置
NAME="mae_deit_pretrain"
MODEL="mae_deit_tiny_patch4"
DATA_PATH="./dataset/cifar10_dataset"
BATCH_SIZE=256
ACCUM=2
EPOCHS=200
WARMUP_EPOCHS=10
BASE_LR=1.5e-4
INPUT_SIZE=32
MASK_RATIO=0.75
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
    --accum_iter)
      ACCUM="$2"
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
    --input_size)
      INPUT_SIZE="$2"
      shift 2
      ;;
    --mask_ratio)
      MASK_RATIO="$2"
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
echo "accum_iter: ${ACCUM}" >> ${PARAMS_FILE}
echo "epochs: ${EPOCHS}" >> ${PARAMS_FILE}
echo "warmup_epochs: ${WARMUP_EPOCHS}" >> ${PARAMS_FILE}
echo "base_lr: ${BASE_LR}" >> ${PARAMS_FILE}
echo "input_size: ${INPUT_SIZE}" >> ${PARAMS_FILE}
echo "mask_ratio: ${MASK_RATIO}" >> ${PARAMS_FILE}
echo "device: ${DEVICE}" >> ${PARAMS_FILE}
echo "log_dir: ${LOG_DIR}" >> ${PARAMS_FILE}
echo "output_dir: ${OUTPUT_DIR}" >> ${PARAMS_FILE}
echo "current_datetime: ${CURRENT_DATETIME}" >> ${PARAMS_FILE}

# 执行 Python 脚本
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
    --device ${DEVICE} \
    --current_datetime ${CURRENT_DATETIME}
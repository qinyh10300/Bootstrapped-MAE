#!/bin/bash
export OMP_NUM_THREADS=4

# 默认 YAML 配置文件路径
DEFAULT_CONFIG_FILE="./experiments/default_mae_deit_tiny.yaml"

# 解析命令行参数
TEMP=$(getopt -o "c:" --long cfg: -n 'mae_train.sh' -- "$@")
if [ $? != 0 ]; then
    echo "Error parsing options."
    exit 1
fi

# 设置解析后的参数
eval set -- "$TEMP"

# 初始化变量
CONFIG_FILE=$DEFAULT_CONFIG_FILE

# 处理命令行参数
while true; do
    case "$1" in
        -c|--cfg)
            CONFIG_FILE="$2"
            echo "Using config file: $CONFIG_FILE"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 加载 YAML 配置  使用yt3.4.3
NUM_GPUS=$(yq r $CONFIG_FILE 'NUM_GPUS')
MODEL=$(yq r $CONFIG_FILE 'MODEL')
DATA_PATH=$(yq r $CONFIG_FILE 'DATA_PATH')
OUTPUT_DIR=$(yq r $CONFIG_FILE 'OUTPUT_DIR')
BATCH_SIZE=$(yq r $CONFIG_FILE 'BATCH_SIZE')
ACCUM=$(yq r $CONFIG_FILE 'ACCUM')
EPOCHS=$(yq r $CONFIG_FILE 'EPOCHS')
WARMUP_EPOCHS=$(yq r $CONFIG_FILE 'WARMUP_EPOCHS')
BASE_LR=$(yq r $CONFIG_FILE 'BASE_LR')
INPUT_SIZE=$(yq r $CONFIG_FILE 'INPUT_SIZE')
MASK_RATIO=$(yq r $CONFIG_FILE 'MASK_RATIO')

# 获取当前日期和时间
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 动态生成日志目录
LOG_DIR="./logs/mae_train/tb_${CURRENT_DATETIME}"

# 运行训练脚本
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
    --log_dir ${LOG_DIR}
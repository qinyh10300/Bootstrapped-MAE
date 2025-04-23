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

echo "Using config file: $CONFIG_FILE"

# 初始化变量，使用 YAML 文件中的默认值
num_gpus=$(yq e '.NUM_GPUS' $CONFIG_FILE)
model=$(yq e '.MODEL' $CONFIG_FILE)
data_path=$(yq e '.DATA_PATH' $CONFIG_FILE)
output_dir=$(yq e '.OUTPUT_DIR' $CONFIG_FILE)
batch_size=$(yq e '.BATCH_SIZE' $CONFIG_FILE)
accum=$(yq e '.ACCUM' $CONFIG_FILE)
epochs=$(yq e '.EPOCHS' $CONFIG_FILE)
warmup_epochs=$(yq e '.WARMUP_EPOCHS' $CONFIG_FILE)
base_lr=$(yq e '.BASE_LR' $CONFIG_FILE)
input_size=$(yq e '.INPUT_SIZE' $CONFIG_FILE)
mask_ratio=$(yq e '.MASK_RATIO' $CONFIG_FILE)

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
#!/bin/bash

# 定义要按顺序运行的 Python 脚本列表
scripts=(
    "experiments/pretrain_accum_tuning.py"
    "experiments/pretrain_half_life_tuning.py"
)

# 按顺序运行列表中的 Python 脚本
for script in "${scripts[@]}"; do
    echo "Running $script..."
    if python "$script"; then
        echo "$script completed successfully!"
    else
        echo "Error occurred while running $script. Skipping to the next script."
    fi
done

echo "All scripts have been executed!"
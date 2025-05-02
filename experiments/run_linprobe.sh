#!/bin/bash

# 定义要按顺序运行的 Python 脚本列表
scripts=(
    "experiments/linprobe_all_bs_step_tuning.py"
    "experiments/linprobe_all_lr_tuning.py"
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
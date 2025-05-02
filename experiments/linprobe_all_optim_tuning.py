import subprocess
import json
import os
from datetime import datetime

# Define the hyperparameters to explore
CKPTS = [
        "ckpts/Bmae_deit_pretrain_pretrain_optim_AdamW/2025-05-02_05-33-17/Bmae-ema-200-0.pth",
        "ckpts/Bmae_deit_pretrain_pretrain_optim_RMSprop/2025-05-02_02-18-48/Bmae-ema-200-0.pth",
        "ckpts/Bmae_deit_pretrain_pretrain_optim_SGD/2025-05-02_03-55-50/Bmae-ema-200-0.pth",
        ]

# Output log file
log_file = "./experiments/hyperparam_results/optim_linprobe.log"

# # Make sure to clear the log file before starting
# if os.path.exists(log_file):
#     assert os.path.getsize(log_file) == 0, f"Log file {log_file} is not empty. Please clear it before running the script."

# Function to parse the log file and extract metrics
def parse_log_file(log_path):
    max_test_acc1 = float('-inf')
    max_test_acc5 = float('-inf')
    min_train_loss = float('inf')
    min_test_loss = float('inf')

    try:
        with open(log_path, "r") as log_file:
            lines = log_file.readlines()
            for line in lines:
                try:
                    # 解析每一行 JSON 数据
                    entry = json.loads(line)
                    # 更新最大 test_acc1 和 test_acc5
                    max_test_acc1 = max(max_test_acc1, entry.get("test_acc1", float('-inf')))
                    max_test_acc5 = max(max_test_acc5, entry.get("test_acc5", float('-inf')))
                    # 更新最小 train_loss 和 test_loss
                    min_train_loss = min(min_train_loss, entry.get("train_loss", float('inf')))
                    min_test_loss = min(min_test_loss, entry.get("test_loss", float('inf')))
                except json.JSONDecodeError:
                    print(f"Error parsing line in {log_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        return None

    return {
        "max_test_acc1": max_test_acc1,
        "max_test_acc5": max_test_acc5,
        "min_train_loss": min_train_loss,
        "min_test_loss": min_test_loss,
    }

# Function to run the training process and capture the last line from log.txt
def run_training(ckpt):
    # Define the output directory based on the hyperparameters
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间并格式化为字符串
    print("current_datetime:", current_datetime)
    ckpt_name = "optim_" + ckpt.split('/')[1].split('_optim_')[1].split('/')[0]
    # print(ckpt_name)
    # exit(0)
    name = f"Bmae_deit_linprobe_{ckpt_name}"
    output_dir = f"./ckpts/{name}/{current_datetime}"
    
    # Run the training script using subprocess
    command = [
        "bash", "bash_scripts/Bmae_eval_linear.sh", 
        "--finetune", str(ckpt),
        "--current_datetime", str(current_datetime),
        "--name", str(name),
        "--device", "cuda:0",
        "--save_frequency", "200"  # 相当于不save checkpoint
    ]
    
    # Run the command and capture the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream the output to the terminal in real-time
    for line in process.stdout:
        print(line, end="")  # Print each line of stdout in real-time
    for line in process.stderr:
        print(line, end="")  # Print each line of stderr in real-time
    
    # Wait for the process to complete
    process.wait()
    
    # Check if the process ran successfully
    if process.returncode != 0:
        print(f"Error occurred with parameters: KPT={ckpt}")
        return None
    
    # Read the last line from the log file
    log_path = os.path.join(output_dir, "log.txt")
    
    # Read and parse the log file
    log_path = os.path.join(output_dir, "log.txt")
    stats = parse_log_file(log_path)
    if stats is not None:
        print(f"Stats for {ckpt}: {stats}")
        return stats
    else:
        print(f"Failed to parse stats for {ckpt}")
        return None

# 确保日志文件的父目录存在
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open(log_file, "a") as log:
    log_current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间并格式化为字符串
    log.write(f"\n\n*****************************************************************\n")
    log.write(f"Start logging linprobe optim tuning, at {log_current_datetime}\n")
    print(f"Start logging linprobe optim tuning, at {log_current_datetime}")

    for ckpt in CKPTS:
        print(f"Running training with CKPT={ckpt}")
        
        stats = run_training(ckpt)
        
        if stats is not None:
            log.write(f"CKPT={ckpt}, MAX_TEST_ACC1={stats['max_test_acc1']}, MAX_TEST_ACC5={stats['max_test_acc5']}, "
                      f"MIN_TRAIN_LOSS={stats['min_train_loss']}, MIN_TEST_LOSS={stats['min_test_loss']}\n")
            print(f"CKPT={ckpt}. Stats: {stats}")
        else:
            log.write(f"ERROR: CKPT={ckpt}, STATUS=FAILED\n")
            print(f"Error with: CKPT={ckpt}. Marking as FAILED.")

    log.write(f"Finish logging linprobe optim tuning, at {log_current_datetime}\n")
    log.write(f"*****************************************************************\n\n")
    print(f"Finish logging linprobe optim tuning, at {log_current_datetime}")
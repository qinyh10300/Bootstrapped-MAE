import subprocess
import json
import os
from datetime import datetime

# Define the hyperparameters to explore
EMA_DECAY = [0.999, 0.99, 0.9]

# Output log file
log_file = "./experiments/hyperparam_results/pretrain_half_life_tuning.log"

# # Make sure to clear the log file before starting
# if os.path.exists(log_file):
#     assert os.path.getsize(log_file) == 0, f"Log file {log_file} is not empty. Please clear it before running the script."

# Function to run the training process and capture the last line from log.txt
def run_training(ema_decay):
    # Define the output directory based on the hyperparameters
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间并格式化为字符串
    print("current_datetime:", current_datetime)
    name = f"Bmae_deit_pretrain_pretrain_ema_decay_{ema_decay}"
    output_dir = f"./ckpts/{name}/{current_datetime}"
    
    # Run the training script using subprocess
    command = [
        "bash", "bash_scripts/Bmae_train.sh", 
        "--ema_decay", str(ema_decay), 
        "--use_ema", 
        "--current_datetime", str(current_datetime),
        "--name", str(name),
        "--device", "cuda:2",
    ]
    
    # Run the command and capture the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Check if the process ran successfully
    if process.returncode != 0:
        print(f"Error occurred with parameters: EMA_DECAY={ema_decay}")
        print(stderr.decode())
        return None
    
    # Read the last line from the log file
    log_path = os.path.join(output_dir, "log.txt")
    
    with open(log_path, "r") as log_file:
        lines = log_file.readlines()
        if lines:
            last_line = lines[-1]
            try:
                # Parse the JSON line to get the training loss
                last_entry = json.loads(last_line)
                train_loss = last_entry.get("train_loss")
                return train_loss
            except json.JSONDecodeError:
                print(f"Error parsing last line of {log_path}")
                return None

# 确保日志文件的父目录存在
log_dir = os.path.dirname(log_file)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open(log_file, "a") as log:
    log_current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间并格式化为字符串
    log.write(f"\n\n*****************************************************************\n")
    log.write(f"Start logging pretrain half-life tuning, at{log_current_datetime}\n")
    print(f"FStart logging pretrain half-life tuning, at{log_current_datetime}")

    # Iterate over all combinations of hyperparameters
    for ema_decay in EMA_DECAY:
        print(f"Running training with EMA_DECAY={ema_decay}")
        
        # Run training and capture the last line's training loss
        train_loss = run_training(ema_decay)
        
        if train_loss is not None:
            log.write(f"EMA_DECAY={ema_decay}, TRAIN_LOSS={train_loss}\n")
            print(f"Finished: EMA_DECAY={ema_decay}, TRAIN_LOSS={train_loss}")
        else:
            log.write(f"ERROR: EMA_DECAY={ema_decay}, STATUS=FAILED\n")
            print(f"Error with: EMA_DECAY={ema_decay}. Marking as FAILED.")
    
    log.write(f"Finish logging pretrain half-life tuning, at{log_current_datetime}\n")
    log.write(f"*****************************************************************\n\n")
    print(f"Finish logging pretrain half-life tuning, at{log_current_datetime}")

import subprocess
import json
import os
from datetime import datetime

# Define the hyperparameters to explore
WEIGHT_DECAY = [0.1, 0.05, 0.01]

# Output log file
log_file = "./experiments/hyperparam_results/pretrain_weight_decay_tuning.log"

# # Make sure to clear the log file before starting
# if os.path.exists(log_file):
#     assert os.path.getsize(log_file) == 0, f"Log file {log_file} is not empty. Please clear it before running the script."

# Function to run the training process and capture the last line from log.txt
def run_training(weight_decay):
    # Define the output directory based on the hyperparameters
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间并格式化为字符串
    print("current_datetime:", current_datetime)
    name = f"Bmae_deit_pretrain_pretrain_weight_decay_{weight_decay}"
    output_dir = f"./ckpts/{name}/{current_datetime}"
    
    # Run the training script using subprocess
    command = [
        "bash", "bash_scripts/Bmae_train.sh", 
        "--weight_decay", str(weight_decay),
        "--current_datetime", str(current_datetime),
        "--name", str(name),
        "--device", "cuda:0",
        "--use_ema",
        "--save_frequency", "200",
        "--bootstrap_steps", "200",
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
        print(f"Error occurred with parameters: WEIGHT_DECAY={weight_decay}")
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
    log.write(f"Start logging pretrain weight_decay tuning, at {log_current_datetime}\n")
    print(f"Start logging pretrain weight_decay tuning, at {log_current_datetime}")

    # Iterate over all combinations of hyperparameters
    for weight_decay in WEIGHT_DECAY:
        print(f"Running training with WEIGHT_DECAY={weight_decay}")
        
        # Run training and capture the last line's training loss
        train_loss = run_training(weight_decay)
        
        if train_loss is not None:
            log.write(f"WEIGHT_DECAY={weight_decay}, TRAIN_LOSS={train_loss}\n")
            print(f"Finished: WEIGHT_DECAY={weight_decay}, TRAIN_LOSS={train_loss}")
        else:
            log.write(f"ERROR: WEIGHT_DECAY={weight_decay}, STATUS=FAILED\n")
            print(f"Error with: WEIGHT_DECAY={weight_decay}. Marking as FAILED.")

    log.write(f"Finish logging pretrain weight_decay tuning, at {log_current_datetime}\n")
    log.write(f"*****************************************************************\n\n")
    print(f"Finish logging pretrain weight_decay tuning, at {log_current_datetime}")
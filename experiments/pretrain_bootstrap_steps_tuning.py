import subprocess
import json
import os
from datetime import datetime

# Define the hyperparameters to explore
BOOTSTRAP_STEPS = [200, 100]
USE_EMA = [True, False]

# Output log file
log_file = "./experiments/hyperparam_results/pretrain_bootstrap_steps_tuning.log"

# # Make sure to clear the log file before starting
# if os.path.exists(log_file):
#     assert os.path.getsize(log_file) == 0, f"Log file {log_file} is not empty. Please clear it before running the script."

# Function to run the training process and capture the last line from log.txt
def run_training(bootstrap_steps, use_ema):
    # Define the output directory based on the hyperparameters
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间并格式化为字符串
    print("current_datetime:", current_datetime)
    name = f"Bmae_deit_pretrain_bootstrap_steps_{bootstrap_steps}_use_ema_{use_ema}"
    output_dir = f"./ckpts/{name}/{current_datetime}"
    
    # Run the training script using subprocess
    command = [
        "bash", "bash_scripts/Bmae_train.sh", 
        "--bootstrap_steps", str(bootstrap_steps),
        "--current_datetime", str(current_datetime),
        "--name", str(name),
        "--device", "cuda:1",
        "--save_frequency", "200"
    ]

    if use_ema:
        command.append("--use_ema")
    
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
        print(f"Error occurred with parameters: BOOTSTRAP_STEPS={bootstrap_steps}, USE_EMA={use_ema}")
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
    log.write(f"Start logging pretrain bs_steps tuning, at {log_current_datetime}\n")
    print(f"Start logging pretrain bs_steps tuning, at {log_current_datetime}")

    for bootstrap_steps in BOOTSTRAP_STEPS:
        for use_ema in USE_EMA:
            if bootstrap_steps == 150 and use_ema == True:
                continue
            print(f"Running training with BOOTSTRAP_STEPS={bootstrap_steps}, USE_EMA={use_ema}")
            
            train_loss = run_training(bootstrap_steps, use_ema)
            
            if train_loss is not None:
                log.write(f"BOOTSTRAP_STEPS={bootstrap_steps}, USE_EMA={use_ema}, TRAIN_LOSS={train_loss}\n")
                print(f"BOOTSTRAP_STEPS={bootstrap_steps}, USE_EMA={use_ema}. TRAIN_LOSS={train_loss}")
            else:
                log.write(f"ERROR: BOOTSTRAP_STEPS={bootstrap_steps}, USE_EMA={use_ema}, STATUS=FAILED\n")
                print(f"Error with: BOOTSTRAP_STEPS={bootstrap_steps}, USE_EMA={use_ema}. Marking as FAILED.")

    log.write(f"Finish logging pretrain bs_step tuning, at {log_current_datetime}\n")
    log.write(f"*****************************************************************\n\n")
    print(f"Finish logging pretrain bs_step tuning, at {log_current_datetime}")
import subprocess
import json
import os
from datetime import datetime

# Define the hyperparameters to explore
CKPTS = [
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_1_use_ema_False/2025-04-30_01-04-55/Bmae-1-199.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_1_use_ema_True/2025-04-29_23-48-56/Bmae-ema-1-199.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_2_use_ema_False/2025-04-30_03-48-26/Bmae-2-99.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_2_use_ema_True/2025-04-30_02-19-41/Bmae-ema-2-99.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_3_use_ema_False/2025-04-30_06-49-46/Bmae-3-65.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_3_use_ema_True/2025-04-30_05-16-33/Bmae-ema-3-65.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_4_use_ema_False/2025-04-30_01-29-18/Bmae-4-49.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_4_use_ema_True/2025-04-29_23-49-28/Bmae-ema-4-49.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_5_use_ema_False/2025-04-30_04-30-49/Bmae-5-39.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_5_use_ema_True/2025-04-30_02-59-18/Bmae-ema-5-39.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_6_use_ema_False/2025-04-30_07-34-05/Bmae-6-32.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_6_use_ema_True/2025-04-30_06-02-02/Bmae-ema-6-32.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_7_use_ema_False/2025-04-30_01-29-02/Bmae-7-27.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_7_use_ema_True/2025-04-29_23-49-56/Bmae-ema-7-27.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_8_use_ema_False/2025-04-30_04-46-36/Bmae-8-24.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_8_use_ema_True/2025-04-30_03-07-37/Bmae-ema-8-24.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_9_use_ema_False/2025-04-30_08-04-23/Bmae-9-21.pth",
        "ckpts/Bmae_deit_pretrain_bootstrap_steps_9_use_ema_True/2025-04-30_06-25-18/Bmae-ema-9-21.pth",
        ]

# Output log file
log_file = "./experiments/hyperparam_results/bootstrap_steps_finetune.log"

# # Make sure to clear the log file before starting
# if os.path.exists(log_file):
#     assert os.path.getsize(log_file) == 0, f"Log file {log_file} is not empty. Please clear it before running the script."

# Function to run the training process and capture the last line from log.txt
def run_training(ckpt):
    # Define the output directory based on the hyperparameters
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 获取当前时间并格式化为字符串
    print("current_datetime:", current_datetime)
    ckpt_name = "bootstrap_steps_" + ckpt.split('/')[1].split('_bootstrap_steps_')[1].split('/')[0]
    # print(ckpt_name)
    # exit(0)
    name = f"Bmae_deit_finetune_{ckpt_name}"
    output_dir = f"./ckpts/{name}/{current_datetime}"
    
    # Run the training script using subprocess
    command = [
        "bash", "bash_scripts/Bmae_eval_finetune.sh", 
        "--finetune", str(ckpt),
        "--current_datetime", str(current_datetime),
        "--name", str(name),
        "--device", "cuda:0",  # finetune必须是cuda:0
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
    log.write(f"Start logging finetune bs_steps tuning, at {log_current_datetime}\n")
    print(f"Start logging finetune bs_steps tuning, at {log_current_datetime}")

    for ckpt in CKPTS:
        print(f"Running training with CKPT={ckpt}")
        
        train_loss = run_training(ckpt)
        
        if train_loss is not None:
            log.write(f"CKPT={ckpt}, TRAIN_LOSS={train_loss}\n")
            print(f"CKPT={ckpt}. TRAIN_LOSS={train_loss}")
        else:
            log.write(f"ERROR: CKPT={ckpt}, STATUS=FAILED\n")
            print(f"Error with: CKPT={ckpt}. Marking as FAILED.")

    log.write(f"Finish logging finetune bs_step tuning, at {log_current_datetime}\n")
    log.write(f"*****************************************************************\n\n")
    print(f"Finish logging finetune bs_step tuning, at {log_current_datetime}")
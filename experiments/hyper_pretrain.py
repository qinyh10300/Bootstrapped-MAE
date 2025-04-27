import subprocess
import json
import os

# Define the hyperparameters to explore
ACCUM_VALUES = [1, 2, 4]
WARMUP_EPOCHS_VALUES = [10]
BASE_LR_VALUES = [1e-4, 1.5e-4]

# Output log file
log_file = "hyperparam_pretrain.log"

# Make sure to clear the log file before starting
if os.path.exists(log_file):
    os.remove(log_file)

# Function to run the training process and capture the last line from log.txt
def run_training(accum, warmup_epochs, base_lr):
    # Define the output directory based on the hyperparameters
    output_dir = f"../ckpts/mae_baseline/pretrain_accum{accum}_warmup{warmup_epochs}_lr{base_lr}"
    
    # Run the training script using subprocess
    command = [
        "torchrun", "--nproc_per_node=8", "../main_pretrain.py", 
        "--world_size", "8", 
        "--model", "mae_vit_tiny_patch4", 
        "--data_path", "../datasets/cifar10_dataset", 
        "--output_dir", output_dir, 
        "--batch_size", "128", 
        "--accum_iter", str(accum), 
        "--epochs", "200", 
        "--warmup_epochs", str(warmup_epochs), 
        "--blr", str(base_lr), 
        "--input_size", "32", 
        "--mask_ratio", "0.75", 
        "--norm_pix_loss", 
        "--log_dir", "../logs/tb"
    ]
    
    # Run the command and capture the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # Check if the process ran successfully
    if process.returncode != 0:
        print(f"Error occurred with parameters: ACCUM={accum}, WARMUP_EPOCHS={warmup_epochs}, BASE_LR={base_lr}")
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

# Open the log file for appending results
with open(log_file, "a") as log:
    # Iterate over all combinations of hyperparameters
    for accum in ACCUM_VALUES:
        for warmup_epochs in WARMUP_EPOCHS_VALUES:
            for base_lr in BASE_LR_VALUES:
                print(f"Running training with ACCUM={accum}, WARMUP_EPOCHS={warmup_epochs}, BASE_LR={base_lr}")
                
                # Run training and capture the last line's training loss
                train_loss = run_training(accum, warmup_epochs, base_lr)
                
                if train_loss is not None:
                    log.write(f"ACCUM={accum}, WARMUP_EPOCHS={warmup_epochs}, BASE_LR={base_lr}, TRAIN_LOSS={train_loss}\n")
                    print(f"Finished: ACCUM={accum}, WARMUP_EPOCHS={warmup_epochs}, BASE_LR={base_lr}, TRAIN_LOSS={train_loss}")
                else:
                    log.write(f"ERROR: ACCUM={accum}, WARMUP_EPOCHS={warmup_epochs}, BASE_LR={base_lr}, STATUS=FAILED\n")
                    print(f"Error with: ACCUM={accum}, WARMUP_EPOCHS={warmup_epochs}, BASE_LR={base_lr}. Marking as FAILED.")

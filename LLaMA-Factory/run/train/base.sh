#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Define the model paths and their corresponding output directories
declare -A models
models=(
#    ["Qwen2.5-0.5B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-0.5B-Instruct"
#    ["Qwen2.5-1.5B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-1.5B-Instruct"
#    ["Qwen2.5-3B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-3B-Instruct"
#    ["Qwen2.5-14B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-14B-Instruct"
    ["Meta-Llama-3.1-8B-Instruct"]="/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct"
    ["Qwen2.5-7B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-7B-Instruct"
    ["Mistral-7B-Instruct-v0.3"]="/mnt/nvme1/hf-model/Mistral-7B-Instruct-v0.3"
)

mkdir -p saves/base
mkdir -p log/base

# Training configuration parameters
dataset="train_base"
max_samples="200000"
per_device_train_batch_size="8"
gradient_accumulation_steps="8"
num_train_epochs="1"
save_steps="10000"

# Loop through each model and run the training task
for model_name in "${!models[@]}"; do
    model_path="${models[$model_name]}"

    # Dynamically set the template based on the model name
    if [[ "$model_name" == *"Qwen"* ]]; then
        template="qwen"
    elif [[ "$model_name" == *"Meta-Llama"* ]]; then
        template="llama3"
    elif [[ "$model_name" == *"Mistral"* ]]; then
        template="mistral"
    else
        template="qwen"  # Default template (can be adjusted if needed)
    fi

    # Set output directories and log files
    output_dir="saves/base/${model_name//./_}/lora/sft"
    log_file="log/base/${model_name//./_}.log"

    echo "Training ${model_name} LoRA-SFT starts..."
    nohup llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
        model_name_or_path="$model_path" \
        dataset="$dataset" \
        template="$template" \
        max_samples="$max_samples" \
        per_device_train_batch_size="$per_device_train_batch_size" \
        gradient_accumulation_steps="$gradient_accumulation_steps" \
        num_train_epochs="$num_train_epochs" \
        save_steps="$save_steps" \
        output_dir="$output_dir" \
    > "$log_file" 2>&1
    echo "${model_name} LoRA-SFT training complete"
done

echo "All model training tasks have been launched."
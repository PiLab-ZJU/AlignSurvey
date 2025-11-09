#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="4,5"

# Create necessary directories
mkdir -p saves/task3
mkdir -p log/task3

# Define common training parameters
TRAIN_SCRIPT="examples/train_lora/llama3_lora_sft.yaml"
MAX_SAMPLES="200000"
PER_DEVICE_BATCH_SIZE="8"
GRADIENT_ACCUMULATION_STEPS="8"
NUM_TRAIN_EPOCHS="3"
SAVE_STEPS="1000"

# Define a function for training
train_model() {
    local model_name=$1       # e.g., "Meta-Llama-3.1-8B-Instruct"
    local model_path=$2       # e.g., "/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct"
    local dataset=$3
    local output_dir=$4
    local log_file=$5

    # Select template based on model name
    local template
    if [[ "$model_name" == *"Qwen"* ]]; then
        template="qwen"
    elif [[ "$model_name" == *"Meta-Llama"* ]]; then
        template="llama3"
    elif [[ "$model_name" == *"Mistral"* ]]; then
        template="mistral"
    else
        template="qwen"
    fi

    echo "$model_name LoRA-SFT training started..."
    nohup llamafactory-cli train "$TRAIN_SCRIPT" \
        model_name_or_path="$model_path" \
        adapter_name_or_path="saves/base/${model_name//./_}/lora/sft" \
        dataset="$dataset" \
        template="$template" \
        max_samples="$MAX_SAMPLES" \
        per_device_train_batch_size="$PER_DEVICE_BATCH_SIZE" \
        gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
        num_train_epochs="$NUM_TRAIN_EPOCHS" \
        save_steps="$SAVE_STEPS" \
        output_dir="$output_dir" \
    > "$log_file" 2>&1
    echo "$model_name LoRA-SFT training finished"
}

# Define models and datasets in an associative array (keys are model names)
declare -A models
#models["Qwen2.5-0.5B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-0.5B-Instruct"
#models["Qwen2.5-1.5B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-1.5B-Instruct"
#models["Qwen2.5-3B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-3B-Instruct"
models["Qwen2.5-7B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-7B-Instruct"
#models["Qwen2.5-14B-Instruct"]="/mnt/nvme1/hf-model/Qwen2.5-14B-Instruct"
models["Meta-Llama-3.1-8B-Instruct"]="/mnt/nvme1/hf-model/Meta-Llama-3.1-8B-Instruct"
models["Mistral-7B-Instruct-v0.3"]="/mnt/nvme1/hf-model/Mistral-7B-Instruct-v0.3"

# Execute training for each model
for model_name in "${!models[@]}"; do
    model_path="${models[$model_name]}"

    train_model "$model_name" "$model_path" \
        "train_attitude" \
        "saves/task3/${model_name//./_}/lora/sft" \
        "log/task3/${model_name//./_}.log"

    train_model "$model_name" "$model_path" \
        "train_attitude_group" \
        "saves/task3/${model_name//./_}_group/lora/sft" \
        "log/task3/${model_name//./_}_group.log"
done
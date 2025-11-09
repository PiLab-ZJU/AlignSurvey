#!/usr/bin/env bash
set -euo pipefail

# Define fields and models
fields=(
  性别
  年龄
  户籍状态
  职业类型
  最高教育程度
  家里人口数
  家庭成员关系
  有房
  有车
  家庭年收入
  家庭月消费主要支出项目
  生活压力程度
)

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

# Set CUDA devices for training
export CUDA_VISIBLE_DEVICES="0,1"

# Create directories for saving and logging
mkdir -p saves/task1
mkdir -p log/task1

# Loop through each field and model for training
for field in "${fields[@]}"; do
  for model_name in "${!models[@]}"; do
    model_path="${models[$model_name]}"

    # Dynamically set the template based on the model
    if [[ "$model_name" == *"Qwen2.5"* ]]; then
      template="qwen"
    elif [[ "$model_name" == *"Meta-Llama"* ]]; then
      template="llama3"
    elif [[ "$model_name" == *"Mistral"* ]]; then
      template="mistral"
    else
      template="qwen"  # Default template (can be adjusted if needed)
    fi

    output_dir="saves/task1/${model_name//./_}_${field}/lora/sft"
    log_file="log/task1/${model_name//./_}_${field}.log"

    echo "[${field}] ▶ ${model_name} LoRA-SFT starts..."
    nohup llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml \
        model_name_or_path="$model_path" \
        adapter_name_or_path="saves/base/${model_name//./_}/lora/sft" \
        dataset="train_${field}" \
        template="$template" \
        max_samples="200000" \
        per_device_train_batch_size="8" \
        gradient_accumulation_steps="8" \
        num_train_epochs="3" \
        save_steps="1000" \
        output_dir="$output_dir" \
    > "$log_file" 2>&1
    echo "[${field}] ◀ ${model_name} LoRA-SFT completed"
  done
done

echo "All model training tasks have been launched."

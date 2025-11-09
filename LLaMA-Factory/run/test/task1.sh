#!/usr/bin/env bash
set -euo pipefail

# CUDA device(s) for inference
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Root dir for base models (can override by exporting HF_MODEL_ROOT)
HF_MODEL_ROOT="${HF_MODEL_ROOT:-/mnt/nvme1/hf-model}"

# Output dirs
mkdir -p generate/task1
mkdir -p log/task1/infer

# Fields
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

# Models that were fine-tuned (their adapter directory equals the training output_dir)
# Keep this in sync with the training script's "models" map and output_dir pattern.
declare -A sft_models
sft_models=(
  ["Meta-Llama-3.1-8B-Instruct"]="$HF_MODEL_ROOT/Meta-Llama-3.1-8B-Instruct"
  ["Qwen2.5-7B-Instruct"]="$HF_MODEL_ROOT/Qwen2.5-7B-Instruct"
  ["Mistral-7B-Instruct-v0.3"]="$HF_MODEL_ROOT/Mistral-7B-Instruct-v0.3"
)

# Zero-shot models (no adapter). You can add/remove as needed.
nosft_models=(
  "gpt-oss-120b"
  "gpt-oss-20b"
  "Qwen3-0.6B"
  "Qwen3-1.7B"
  "Qwen3-8B"
  "Qwen3-14B"
  "Qwen3-32B"
  "Qwen3-72B"
  "Qwen2.5-0.5B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-3B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
  "Qwen2.5-32B-Instruct"
  "Qwen2.5-72B-Instruct"
  "DeepSeek-R1-Distill-Qwen-14B"
  "Meta-Llama-3.1-8B-Instruct"
  "Mistral-7B-Instruct-v0.3"
)

# Utilities
sanitize() {
  local s="$1"
  s="${s//\//_}"
  s="${s//./_}"
  s="${s// /_}"
  echo "$s"
}

get_template() {
  local name="$1"
  if [[ "$name" == gpt-oss-* ]]; then
    echo "gpt"
  elif [[ "$name" == Qwen3-* ]]; then
    echo "qwen3"
  elif [[ "$name" == Qwen2.5-* ]]; then
    echo "qwen"
  elif [[ "$name" == *"DeepSeek-R1"* ]]; then
    echo "deepseekr1"
  elif [[ "$name" == *"Meta-Llama-3.1"* ]]; then
    echo "llama3"
  elif [[ "$name" == *"Mistral"* ]]; then
    echo "mistral"
  else
    # Fallback
    echo "qwen"
  fi
}

run_infer() {
  local field="$1"
  local model_name="$2"
  local model_path="$3"
  local template="$4"
  local save_name="$5"
  local log_file="$6"
  local adapter_path="${7:-}"

  echo "[${field}] ▶ ${model_name} begin..."
  if [[ -n "${adapter_path}" ]]; then
    python scripts/vllm_infer.py \
      --model_name_or_path "$model_path" \
      --adapter_name_or_path "$adapter_path" \
      --template "$template" \
      --dataset "test_${field}" \
      --save_name "$save_name" \
      > "$log_file" 2>&1
  else
    python scripts/vllm_infer.py \
      --model_name_or_path "$model_path" \
      --template "$template" \
      --dataset "test_${field}" \
      --save_name "$save_name" \
      > "$log_file" 2>&1
  fi
  echo "[${field}] ◀ ${model_name} end"
}

# Main loops
for field in "${fields[@]}"; do
  # 1) Inference with SFT adapters (adapter_name_or_path = training output_dir)
  for model_name in "${!sft_models[@]}"; do
    model_path="${sft_models[$model_name]}"
    template="$(get_template "$model_name")"
    model_tag="$(sanitize "$model_name")"
    adapter_dir="saves/task1/${model_tag}_${field}/lora/sft"   # Must match training output_dir
    save_file="generate/task1/sft_${model_tag}_${field}.jsonl"
    log_file="log/task1/infer/sft_${model_tag}_${field}.log"

    if [[ ! -d "$adapter_dir" ]]; then
      echo "[${field}] Adapter directory not found: $adapter_dir, skipping SFT inference for ${model_name}"
      continue
    fi

    run_infer "$field" "$model_name" "$model_path" "$template" "$save_file" "$log_file" "$adapter_dir"
  done

  # 2) Zero-shot inference (no adapter)
  for model_name in "${nosft_models[@]}"; do
    template="$(get_template "$model_name")"
    model_tag="$(sanitize "$model_name")"
    model_path="$HF_MODEL_ROOT/$model_name"
    save_file="generate/task1/nosft_${model_tag}_${field}.jsonl"
    log_file="log/task1/infer/nosft_${model_tag}_${field}.log"

    run_infer "$field" "$model_name" "$model_path" "$template" "$save_file" "$log_file"
  done
done

echo "All inference tasks have been completed/submitted."
#!/usr/bin/env bash
set -euo pipefail

# GPUs for inference
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Base model root (override with: export HF_MODEL_ROOT=/path/to/models)
HF_MODEL_ROOT="${HF_MODEL_ROOT:-/mnt/nvme1/hf-model}"

# Dirs
mkdir -p generate/task3 generate/task3_group
mkdir -p log/task3/infer log/task3_group/infer

# Zero-shot models (no adapter)
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

# Models with SFT adapters; base model paths
# Must match training outputs:
# - train_attitude:      saves/task3/${model_tag}/lora/sft
# - train_attitude_group: saves/task3/${model_tag}_group/lora/sft
declare -A sft_models=(
  ["Meta-Llama-3.1-8B-Instruct"]="$HF_MODEL_ROOT/Meta-Llama-3.1-8B-Instruct"
  ["Qwen2.5-7B-Instruct"]="$HF_MODEL_ROOT/Qwen2.5-7B-Instruct"
  ["Mistral-7B-Instruct-v0.3"]="$HF_MODEL_ROOT/Mistral-7B-Instruct-v0.3"
)

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
    echo "qwen"
  fi
}

run_infer() {
  local model_name="$1"
  local model_path="$2"
  local template="$3"
  local dataset="$4"
  local save_name="$5"
  local log_file="$6"
  local adapter_path="${7:-}"

  echo "${model_name} 推理开始..."
  if [[ -n "$adapter_path" ]]; then
    python scripts/vllm_infer.py \
      --model_name_or_path "$model_path" \
      --adapter_name_or_path "$adapter_path" \
      --template "$template" \
      --dataset "$dataset" \
      --save_name "$save_name" \
      > "$log_file" 2>&1
  else
    python scripts/vllm_infer.py \
      --model_name_or_path "$model_path" \
      --template "$template" \
      --dataset "$dataset" \
      --save_name "$save_name" \
      > "$log_file" 2>&1
  fi
  echo "${model_name} 推理完成"
}

# Datasets: base and group
datasets=("test_attitude" "test_attitude_group")

for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "test_attitude_group" ]]; then
    gen_dir="generate/task3_group"
    log_dir="log/task3_group/infer"
    group_suffix="_group"
  else
    gen_dir="generate/task3"
    log_dir="log/task3/infer"
    group_suffix=""
  fi

  mkdir -p "$gen_dir" "$log_dir"

  # 1) SFT inference (adapter_name_or_path must equal training output_dir)
  for model_name in "${!sft_models[@]}"; do
    model_path="${sft_models[$model_name]}"
    template="$(get_template "$model_name")"
    model_tag="$(sanitize "$model_name")"

    adapter_dir="saves/task3/${model_tag}${group_suffix}/lora/sft"
    save_file="${gen_dir}/sft_${model_tag}.jsonl"
    log_file="${log_dir}/sft_${model_tag}.log"

    if [[ ! -d "$adapter_dir" ]]; then
      echo "Adapter directory not found: $adapter_dir, skipping SFT inference for ${model_name} (dataset=${dataset})"
      continue
    fi

    run_infer "$model_name" "$model_path" "$template" "$dataset" "$save_file" "$log_file" "$adapter_dir"
  done

  # 2) Zero-shot inference (no adapter)
  for model_name in "${nosft_models[@]}"; do
    template="$(get_template "$model_name")"
    model_tag="$(sanitize "$model_name")"
    model_path="$HF_MODEL_ROOT/$model_name"
    save_file="${gen_dir}/nosft_${model_tag}.jsonl"
    log_file="${log_dir}/nosft_${model_tag}.log"

    run_infer "$model_name" "$model_path" "$template" "$dataset" "$save_file" "$log_file"
  done
done

echo "All task3 inference tasks have been completed/submitted."
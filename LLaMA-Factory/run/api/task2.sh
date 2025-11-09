#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

mkdir -p generate/task2
mkdir -p log/task2/infer/

echo "gpt5 推理开始"
nohup python -u api/llm_api.py \
  --model "gpt-5" \
  --input "../data/enitre_pipeline/task2/dialouge_test.json" \
  --output "generate/task2/gpt5_${field}.jsonl" \
  > "log/task2/infer/gpt5_${field}.log" 2>&1 &
echo "gpt5 推理提交"
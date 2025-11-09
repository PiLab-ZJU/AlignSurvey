#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

mkdir -p generate/task3
mkdir -p log/task3/infer/
mkdir -p generate/task3_group
mkdir -p log/task3_group/infer/

echo "gpt5 推理开始"
nohup python -u api/llm_api.py \
  --model "gpt-5" \
  --input "../data/enitre_pipeline/task3/individual/test_attitude.json" \
  --output "generate/task3/gpt5.jsonl" \
  > "log/task3/infer/gpt5.log" 2>&1 &
echo "gpt5 推理提交"

echo "gpt5 group 推理开始"
nohup python -u api/llm_api.py \
  --model "gpt-5" \
  --input "../data/enitre_pipeline/task3/group/test_attitude_group.json" \
  --output "generate/task3_group/gpt5.jsonl" \
  > "log/task3_group/infer/gpt5.log" 2>&1 &
echo "gpt5 group 推理提交"
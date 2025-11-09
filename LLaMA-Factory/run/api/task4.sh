#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

mkdir -p generate/task4
mkdir -p log/task4/infer/
mkdir -p generate/task4_group
mkdir -p log/task4_group/infer/

echo "gpt5 推理开始"
nohup python -u api/llm_api.py \
  --model "gpt-5" \
  --input "../data/enitre_pipeline/task4/ase/individual/ase_test.json" \
  --output "generate/task4/gpt5.jsonl" \
  > "log/task4/infer/gpt5.log" 2>&1 &
echo "gpt5 推理提交"

echo "gpt5 group 推理开始"
nohup python -u api/llm_api.py \
  --model "gpt-5" \
  --input "../data/enitre_pipeline/task4/ase/group/ase_test_group.json" \
  --output "generate/task4_group/gpt5.jsonl" \
  > "log/task4_group/infer/gpt5.log" 2>&1 &
echo "gpt5 group 推理提交"

#echo "gpt4o 推理开始"
#nohup python -u api/llm_api.py \
#  --model "gpt-4o" \
#  --input "../data/enitre_pipeline/task4/ase/individual/ase_test.json" \
#  --output "generate/task4/gpt4o.jsonl" \
#  > "log/task4/infer/gpt4o.log" 2>&1 &
#echo "gpt4o 推理提交"
#
#echo "gpt4o group 推理开始"
#nohup python -u api/llm_api.py \
#  --model "gpt-4o" \
#  --input "../data/enitre_pipeline/task4/ase/group/ase_test_group.json" \
#  --output "generate/task4_group/gpt4o.jsonl" \
#  > "log/task4_group/infer/gpt4o.log" 2>&1 &
#echo "gpt4o group 推理提交"
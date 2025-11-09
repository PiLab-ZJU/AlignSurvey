#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

mkdir -p generate/task1
mkdir -p log/task1/infer/

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

for field in "${fields[@]}"; do
  echo "[${field}] ◀ gpt 推理开始"
  nohup python -u api/llm_api.py \
    --model "gpt-5" \
    --input "../data/enitre_pipeline/task1/back/test_${field}.json" \
    --output "generate/task1/gpt5_${field}.jsonl" \
    > "log/task1/infer/gpt5_${field}.log" 2>&1 &
  echo "[${field}] ◀ gpt 推理提交"
done

wait
echo "全部推理完成"
import json
import pandas as pd
import numpy as np
import re
import sys
import os

temple = """**模拟访谈回答评估标准**

**评分维度**
**1. 对话自然度**: 评估访谈回答是否以自然语言表达、与提问形成合理衔接，是否呈现出真实人物交流的语气和节奏，应避免模板化、条列式回答。**若回答结构过于工整，应适当降低本项得分**。
**2. 风格匹配度**: 衡量回答是否与标准回答一致，包括语言风格、思维方式、关注重点等。
**3. 访谈一致性**: 判断回答是否准确表达与问题相关的核心观点，是否覆盖标准答案所体现的主要立场或逻辑。**如回答中表达了标准答案未包含的判断，应适当降低本项得分**。

**评分标准**
基于以上维度对模拟回答进行评分，每个维度使用 1-5 分制
- **5:** 完全符合评价标准，表达准确、自然、完整
- **4:** 基本符合标准，存在轻微不足或偏差
- **3:** 部分符合标准，存在明显问题或缺失
- **2:** 仅少部分符合标准，问题较多、影响理解
- **1:** 完全不符合标准，结构混乱或严重偏离主题

**背景与内容**
{prompt}

**标准答案**
{label}

**模拟回答**
{predict}

**输出格式**
{{ "对话自然度": {{ "评分": "X", "评语": "……" }}, "风格匹配度": {{ "评分": "Y", "评语": "……" }}, "访谈一致性": {{ "评分": "Z", "评语": "……" }} }}
"""

# 定义输入和输出文件夹
input_folder = "generate/dialogue/"
output_folder = "task/interview_generate/update_eval/prompt/"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有 .jsonl 文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jsonl"):
        prompt_list = []
        input_file_path = os.path.join(input_folder, filename)

        # 读取每个 .jsonl 文件
        with open(input_file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                instruction = temple.format(prompt=data["prompt"], label=data["label"], predict=data["predict"])
                prompt = {"instruction": instruction, "label": data["label"], "predict": data["predict"]}
                prompt_list.append(prompt)

        # 构造输出文件路径
        output_filename = filename.replace(".jsonl", "_prompt.jsonl")
        output_file_path = os.path.join(output_folder, output_filename)

        # 将结果写入对应的输出文件
        with open(output_file_path, "w") as f:
            json.dump(prompt_list, f, indent=4, ensure_ascii=False)

print("所有 .jsonl 文件已处理完成！")
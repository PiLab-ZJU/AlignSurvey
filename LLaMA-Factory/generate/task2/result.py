import json
import os
import pandas as pd
import re

def extract_json_from_markdown(text):
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()

def extract_scores(predict_text):
    try:
        cleaned = extract_json_from_markdown(predict_text)
        if cleaned.strip().startswith("{"):
            predict_dict = json.loads(cleaned)
            view = predict_dict.get("对话自然度", {}).get("评分")
            tone = predict_dict.get("风格匹配度", {}).get("评分")
            align = predict_dict.get("访谈一致性",{}).get("评分")
            return int(view) if view else None, int(tone) if tone else None, int(align) if tone else None
    except:
        pass

    # 匹配 Markdown 自然语言格式
    view_match = re.search(r"对话自然度.*?[：:]\s*\*+\s*评分\s*\*+[:：]?\s*(\d)", predict_text, re.DOTALL)
    tone_match = re.search(r"风格匹配度.*?[：:]\s*\*+\s*评分\s*\*+[:：]?\s*(\d)", predict_text, re.DOTALL)
    align_match = re.search(r"访谈一致性.*?[：:]\s*\*+\s*评分\s*\*+[:：]?\s*(\d)", predict_text, re.DOTALL)

    view_score = int(view_match.group(1)) if view_match else None
    tone_score = int(tone_match.group(1)) if tone_match else None
    align_score = int(align_match.group(1)) if align_match else None

    return view_score, tone_score, align_score

jsonl_files = [f for f in os.listdir("") if f.endswith(".jsonl")]

summary_data = []

for input_filename in jsonl_files:
    print(f"处理文件: {input_filename}")
    views = []
    tone = []
    align = []

    with open(input_filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            predict = data.get("predict")
            view_score, tone_score, align_score = extract_scores(predict)

            if view_score is not None:
                views.append(int(view_score))
            if tone_score is not None:
                tone.append(int(tone_score))
            if align_score is not None:
                align.append(int(align_score))

    # 汇总均值
    summary_row = {"模型名": input_filename}
    summary_row["对话自然度"] = round(sum(views) / len(views), 2) if views else "未提取"
    summary_row["风格匹配度"] = round(sum(tone) / len(tone), 2) if tone else "未提取"
    summary_row["访谈一致性"] = round(sum(align) / len(align), 2) if align else "未提取"

    summary_data.append(summary_row)

# 保存总表
df = pd.DataFrame(summary_data)
df.to_excel("summary.xlsx", index=False)

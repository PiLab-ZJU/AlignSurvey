import json
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import re

def process_jsonl_files(folder_path="."):
    """处理指定文件夹中的所有JSONL文件，计算评估指标并导出结果"""
    all_results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl") or filename.endswith(".json"):
            file_results = process_single_file(os.path.join(folder_path, filename))
            if file_results:
                all_results.append({
                    "filename": filename,
                    "correct_count": file_results["correct_count"],
                    "total_count": file_results["total_count"],
                    "accuracy": file_results["accuracy"],
                    # 宏平均指标
                    "precision_macro": file_results["precision_macro"],
                    "recall_macro": file_results["recall_macro"],
                    "f1_macro": file_results["f1_macro"],
                    # 加权平均指标
                    "precision_weighted": file_results["precision_weighted"],
                    "recall_weighted": file_results["recall_weighted"],
                    "f1_weighted": file_results["f1_weighted"],
                    # 选项维度指标
                    "acc_A": file_results["accuracy_by_option"].get("A", 0),
                    "acc_B": file_results["accuracy_by_option"].get("B", 0),
                    "acc_C": file_results["accuracy_by_option"].get("C", 0),
                    "cnt_A": file_results["count_by_option"].get("A", 0),
                    "cnt_B": file_results["count_by_option"].get("B", 0),
                    "cnt_C": file_results["count_by_option"].get("C", 0),
                })

    if all_results:
        export_to_xlsx(all_results, os.path.join(folder_path, "evaluation_results.xlsx"))
        print("结果已成功导出到 evaluation_results.xlsx")
    else:
        print("没有找到有效的JSONL文件或评估结果")

def extract_option(text):
    if len(text) >= 2:
        first_char = text[0].upper()
        if first_char.isalpha() and text[1] in ('.', ' '):
            return first_char, text[2:].strip()
    match = re.match(r'\(([A-Za-z])\)\s*(.*)', text)
    if match:
        return match.group(1).upper(), match.group(2).strip()
    match = re.search(r'([A-Za-z])[.\)\s]+(.*)', text)
    if match:
        return match.group(1).upper(), match.group(2).strip()
    return None, text

def compute_confusion(true_labels, predictions):
    classes = sorted(set(true_labels + predictions))
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    for t, p in zip(true_labels, predictions):
        if p == t:
            tp[p] += 1
        else:
            if p:
                fp[p] += 1
            if t:
                fn[t] += 1
    return tp, fp, fn

def compute_macro(tp, fp, fn):
    precisions, recalls, f1s = [], [], []
    for c in tp:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] > 0 else 0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] > 0 else 0
        f = 2 * p * r / (p + r) if p + r > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls), sum(f1s) / len(f1s)

def process_single_file(file_path):
    true_labels, predictions = [], []
    correct_count = total_count = 0
    correct_by_option = defaultdict(int)
    count_by_option = defaultdict(int)

    with open(file_path, "r", encoding="utf-8") as f:
        print(file_path)
        for line in f:
            result = json.loads(line)
            label = result.get("label", result.get("output", "")).strip()
            predict = result.get("predict", "").strip()
            if "<think>" in predict:
                predict = predict.split("</think>")[-1].strip()
            if "<think>" in label:
                label = label.split("</think>")[-1].strip()
            if "assistantfinal" in predict:
                predict = predict.split("assistantfinal")[-1].strip()
            if "assistantfinal" in label:
                label = label.split("analysisassistantfinal")[-1].strip()
            if not label or "不了解" in label:
                continue
            total_count += 1
            t_opt, t_lbl = extract_option(label)
            p_opt, p_lbl = extract_option(predict)
            label_str = (t_opt + t_lbl) if t_opt else t_lbl
            predict_str = (p_opt + p_lbl) if p_opt else p_lbl
            true_labels.append(label_str)
            predictions.append(predict_str)
            if t_opt:
                count_by_option[t_opt] += 1
            if label_str == predict_str:
                correct_count += 1
                if t_opt:
                    correct_by_option[t_opt] += 1

    accuracy = correct_count / total_count if total_count else 0
    precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)

    accuracy_by_option = {opt: (correct_by_option.get(opt, 0) / count_by_option.get(opt, 1))
                          for opt in ["A", "B", "C"]}

    print(f"\n文件: {os.path.basename(file_path)}")
    print(f"准确率: {accuracy:.2%}")
    print(f"宏 平均 -> 精准率: {precision_macro:.2%}, 召回率: {recall_macro:.2%}, F1: {f1_macro:.2%}")
    print(f"加权 平均 -> 精准率: {precision_weighted:.2%}, 召回率: {recall_weighted:.2%}, F1: {f1_weighted:.2%}")
    for opt in ["A", "B", "C"]:
        print(f"选项{opt} 准确率: {accuracy_by_option[opt]:.2%} (样本数 {count_by_option.get(opt,0)})")

    return {
        "correct_count": correct_count,
        "total_count": total_count,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "accuracy_by_option": accuracy_by_option,
        "count_by_option": count_by_option
    }

def calculate_precision(true_labels, predictions):
    true_positives = sum(1 for t, p in zip(true_labels, predictions) if p and p in t)
    predicted_positives = sum(1 for p in predictions if p)
    return true_positives / predicted_positives if predicted_positives else 0

def calculate_recall(true_labels, predictions):
    true_positives = sum(1 for t, p in zip(true_labels, predictions) if p and p in t)
    return true_positives / len(true_labels) if true_labels else 0

def export_to_xlsx(results, output_path):
    df = pd.DataFrame(results)
    df.columns = [
        "文件名", "正确预测数", "总样本数", "整体准确率",
        "精准率（宏）", "召回率（宏）", "F1分数（宏）",
        "精准率（加权）", "召回率（加权）", "F1分数（加权）",
        "选项A准确率", "选项B准确率", "选项C准确率",
        "选项A数量", "选项B数量", "选项C数量"
    ]

    percent_cols = [
        "整体准确率",
        "精准率（宏）", "召回率（宏）", "F1分数（宏）",
        "精准率（加权）", "召回率（加权）", "F1分数（加权）",
        "选项A准确率", "选项B准确率", "选项C准确率"
    ]
    for col in percent_cols:
        df[col] = df[col].apply(lambda x: f"{x:.2%}")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="评估结果", index=False)
        sheet = writer.sheets["评估结果"]
        for i, col in enumerate(df.columns):
            width = max(df[col].astype(str).map(len).max(), len(col)) + 2
            sheet.column_dimensions[chr(65 + i)].width = width

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JSONL评估工具 - 计算准确率、精准率、召回率和F1分数（宏/加权）")
    parser.add_argument("-d", "--directory", default=".", help="包含JSONL文件的目录路径")
    args = parser.parse_args()
    process_jsonl_files(args.directory)

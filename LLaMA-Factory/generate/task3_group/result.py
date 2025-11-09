import os
import glob
import json
import re
from scipy.stats import wasserstein_distance
import pandas as pd


def extract_option_mapping(instruction: str):
    """从instruction中提取选项标签到标准类别的映射关系"""
    mapping = {}
    # 匹配选项列表的模式，如"A. 满意\nB. 一般\nC. 不满意"
    pattern = r'选项：\s*([\s\S]*?)(?:\n\n|$)'
    match = re.search(pattern, instruction)

    if match:
        options_text = match.group(1)
        # 匹配每个选项，如"A. 满意"
        option_pattern = r'([A-Z])\s*\.\s*([^\n]+)'
        for option_match in re.finditer(option_pattern, options_text):
            label, category = option_match.groups()
            mapping[label] = category.strip()

    return mapping


def parse_distribution(text: str, option_mapping: dict):
    """
    从文本中解析出 {类别: 百分比(0–1)} 的字典
    支持格式：
    1. "A.满意: 45%\nB.一般: 27%\nC.不满意: 27%"
    2. "满意: 45%\n一般: 27%\n不满意: 27%"
    3. JSON格式：{"满意": "45%", "一般": "27%", "不满意": "27%"}
    """
    # 首先尝试JSON解析
    try:
        data = json.loads(text)
        # 如果JSON解析成功，检查格式
        if all(isinstance(v, str) and '%' in v for k, v in data.items()):
            return {k: float(v.strip('%')) / 100.0 for k, v in data.items()}
        elif all(isinstance(v, (int, float)) for k, v in data.items()):
            return {k: float(v) for k, v in data.items()}
    except:
        pass

    # 处理非JSON格式
    dist = {}

    # 处理类似 "A.满意: 45%" 的格式
    pattern1 = r'\s*([A-Z])\s*\.\s*([^:：]+)[:：]\s*([\d.]+)%'
    for m in re.finditer(pattern1, text):
        label, category, value = m.groups()
        std_category = option_mapping.get(label, category.strip())
        dist[std_category] = float(value) / 100.0

    # 如果没找到A.满意这种格式，尝试直接解析"满意: 45%"格式
    if not dist:
        pattern2 = r'\s*([^:：]+)[:：]\s*([\d.]+)%'
        for m in re.finditer(pattern2, text):
            category, value = m.groups()
            dist[category.strip()] = float(value) / 100.0

    return dist


def load_records(path: str):
    """支持 .json 和 .jsonl，按行读取 JSON 对象"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_wd_for_record(rec: dict):
    """计算单个记录的Wasserstein距离"""
    try:
        lbl = rec["label"]
    except:
        lbl = rec["output"]

    # 提取选项映射
    instruction = rec.get("instruction") or rec.get("prompt", "")
    option_mapping = extract_option_mapping(instruction)
    if not option_mapping:
        # 使用默认映射
        option_mapping = {
            'A': '积极',
            'B': '中立',
            'C': '消极',
        }
        print("警告: 未能从instruction中提取选项映射，使用默认映射")

    # 解析真实分布
    true_dist = parse_distribution(str(lbl), option_mapping)

    print(f"真实分布: {true_dist}")

    # 解析模型预测分布
    pred_text = rec.get('predict') or rec.get('prediction') or ""
    pred_dist = parse_distribution(pred_text, option_mapping)

    # 处理无预测分布的情况
    if not pred_dist:
        print("警告: 未能从预测文本中解析出任何分布")
        return max_possible_wd(true_dist)

    print(f"预测分布: {pred_dist}")

    # 对齐类别并计算Wasserstein距离
    cats = sorted(set(true_dist) | set(pred_dist))
    positions = list(range(len(cats)))
    w_true = [true_dist.get(c, 0.0) for c in cats]
    w_pred = [pred_dist.get(c, 0.0) for c in cats]

    print(f"类别顺序: {cats}")
    print(f"预测权重: {w_pred}")
    print(f"真实权重: {w_true}")

    # 检查权重总和
    sum_true = sum(w_true)
    sum_pred = sum(w_pred)

    if sum_true <= 0:
        print("警告: 真实分布的权重总和非正")
        return float('nan')

    if sum_pred <= 0:
        print("警告: 预测分布的权重总和非正")
        return max_possible_wd(true_dist)

    # 计算Wasserstein距离
    return wasserstein_distance(positions, positions, w_true, w_pred)


def max_possible_wd(true_dist: dict) -> float:
    """计算给定真实分布下的最大可能Wasserstein距离"""
    n = len(true_dist)
    return n - 1 if n > 1 else 0.0


def evaluate_folder(folder: str):
    """遍历文件夹下所有JSON/JSONL文件，计算平均WD"""
    results = {}
    patterns = [os.path.join(folder, '*.json'), os.path.join(folder, '*.jsonl')]
    for pattern in patterns:
        for path in glob.glob(pattern):
            recs = load_records(path)
            wds = [compute_wd_for_record(rec) for rec in recs]
            valid_wds = [wd for wd in wds if not pd.isna(wd)]
            avg_wd = sum(valid_wds) / len(valid_wds) if valid_wds else float('nan')
            results[os.path.basename(path)] = avg_wd
    return results


if __name__ == '__main__':
    current_folder = os.getcwd()
    res = evaluate_folder(current_folder)
    print("文件名\t平均 WD")
    for fname, wd in res.items():
        print(f"{fname}\t{wd:.4f}")
import json
import re
from collections import defaultdict
import jieba
import numpy as np
import pandas as pd
from datetime import datetime
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score


def decode_unicode_escapes(obj):
    """递归解码Unicode转义字符"""
    if isinstance(obj, str):
        def repl(m):
            return chr(int(m.group(1), 16))

        return re.sub(r'\\u([0-9a-fA-F]{4})', repl, obj)
    elif isinstance(obj, list):
        return [decode_unicode_escapes(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: decode_unicode_escapes(v) for k, v in obj.items()}
    else:
        return obj

def fix_invalid_unicode_escapes(text: str) -> str:
    """修复无效的Unicode转义序列"""
    # \u 后面不是 4 位 0-9a-f 就替换为 \\u
    return re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', text)


def extract_content_after_think(text: str) -> str:
    """处理<think>标签，提取</think>之后的内容"""
    # 先解码Unicode转义字符
    text = decode_unicode_escapes(text)

    # 检查是否包含</think>标签
    think_end_pattern = r'</think>\s*\n\s*\n'
    match = re.search(think_end_pattern, text)

    if match:
        # 提取</think>\n\n之后的内容
        content_after_think = text[match.end():]
        return content_after_think.strip()
    else:
        # 如果没有</think>标签，返回原内容
        return text.strip()


def clean_markdown_json(text: str) -> str:
    """清理markdown代码块标记"""
    text = re.sub(r'^```(json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def extract_json_from_text(text: str) -> dict:
    """从文本中提取并解析JSON"""
    # 1. 清理markdown标记

    # 2. 修复无效Unicode转义
    cleaned = fix_invalid_unicode_escapes(text)

    # 3. 解码Unicode转义字符
    cleaned = decode_unicode_escapes(cleaned)
    cleaned = extract_content_after_think(cleaned)
    cleaned = clean_markdown_json(cleaned)

    try:
        # 尝试解析JSON
        parsed_json = json.loads(cleaned.strip())
        # 处理reason字段（如果是列表则合并为字符串）
        if isinstance(parsed_json.get("reason"), list):
            parsed_json["reason"] = " ".join(parsed_json["reason"])
        return parsed_json
    except json.JSONDecodeError:
        try:
            parsed_json = json.loads(cleaned + "\"}")
            if isinstance(parsed_json.get("reason"), list):
                parsed_json["reason"] = " ".join(parsed_json["reason"])
            return parsed_json
        except:
            print(f"JSON解析失败: {cleaned[:20]}...")
            return {"predict": "错误", "reason": " "}


def calculate_text_similarity(pred_reason: str, label_reason: str) -> dict:
    def clean_and_tokenize_chinese(text):
        # 移除标点符号和数字，保留中文和英文
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        # 使用jieba进行中文分词
        return set(jieba.cut(text))

    # 获取词汇集合
    pred_tokens = clean_and_tokenize_chinese(pred_reason)
    label_tokens = clean_and_tokenize_chinese(label_reason)

    # 计算词汇重叠
    common_words = pred_tokens.intersection(label_tokens)
    vocab_overlap = len(common_words)

    # 计算Jaccard相似度
    union_words = pred_tokens.union(label_tokens)
    jaccard_sim = len(common_words) / len(union_words) if union_words else 0

    # 计算余弦相似度
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b|[\u4e00-\u9fff]+')
    try:
        vectorizer.fit([pred_reason, label_reason])
        vectors = vectorizer.transform([pred_reason, label_reason])
        cosine_sim = cosine_similarity(vectors)[0, 1]
    except:
        cosine_sim = 0

    return {
        "vocabulary_overlap": vocab_overlap,
        "common_words": list(common_words),
        "jaccard_similarity": jaccard_sim,
        "cosine_similarity": cosine_sim
    }


def analyze_single_prediction(prediction_str: str, label_str: str) -> dict:
    """分析单个预测结果"""
    # 步骤1: 解析预测结果JSON
    if "assistantfinal" in prediction_str:
        prediction_str = prediction_str.split("assistantfinal")[-1].strip()
    if "assistantfinal" in label_str:
        label_strs = label_str.split("analysisassistantfinal")[-1].strip()

    prediction_json = extract_json_from_text(prediction_str)
    prediction = prediction_json["predict"]
    prediction_reason = prediction_json["reason"]

    # 步骤2: 处理标签（先处理<think>标签，再解析JSON）
    processed_label = extract_content_after_think(label_str)
    label_json = extract_json_from_text(processed_label)
    label = label_json['predict']
    label_reason = label_json['reason']

    # 步骤3: 计算相似度
    similarities = calculate_text_similarity(prediction_reason, label_reason)

    # 步骤4: 判断预测是否正确
    is_correct = (prediction == label)

    return {
        'prediction': prediction,
        'prediction_reason': prediction_reason,
        'label': label,
        'label_reason': label_reason,
        'is_correct': is_correct,
        'similarities': similarities
    }


def analyze_predictions_file(file_path: str) -> dict:
    """分析单个预测结果文件"""
    results = []
    total = 0
    correct = 0

    # 用于计算整体指标
    all_predicted_labels = []
    all_true_labels = []

    # 相似度累计
    total_vocab_overlap = 0
    total_jaccard_sim = 0
    total_cosine_sim = 0

    # 按类别统计
    categories = defaultdict(lambda: {
        'total': 0, 'correct': 0, 'errors': 0,
        'vocab_overlap': 0, 'jaccard_sim': 0, 'cosine_sim': 0,
        'error_details': []  # 新增：记录错误详情
    })

    # 计算分类指标
    category_true_positives = defaultdict(int)
    category_false_positives = defaultdict(int)
    category_false_negatives = defaultdict(int)
    all_categories = set()

    mismatches = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)

                # 分析单个预测
                try:
                    analysis = analyze_single_prediction(record["predict"], record['label'])
                except:
                    analysis = analyze_single_prediction(record["predict"], record['output'])

                prediction = analysis['prediction']
                label = analysis['label']

                # 添加到类别集合
                all_categories.add(label)
                all_categories.add(prediction)

                total += 1
                all_predicted_labels.append(prediction)
                all_true_labels.append(label)

                # 累计相似度指标
                similarities = analysis['similarities']
                total_vocab_overlap += similarities['vocabulary_overlap']
                total_jaccard_sim += similarities['jaccard_similarity']
                total_cosine_sim += similarities['cosine_similarity']

                # 按类别统计
                categories[label]['total'] += 1
                categories[label]['vocab_overlap'] += similarities['vocabulary_overlap']
                categories[label]['jaccard_sim'] += similarities['jaccard_similarity']
                categories[label]['cosine_sim'] += similarities['cosine_similarity']

                if analysis['is_correct']:
                    correct += 1
                    categories[label]['correct'] += 1
                    category_true_positives[label] += 1
                else:
                    # 错误统计
                    categories[label]['errors'] += 1
                    categories[label]['error_details'].append({
                        'line_number': line_num,
                        'predicted_as': prediction,
                        'prediction_reason': analysis['prediction_reason'],
                        'label_reason': analysis['label_reason']
                    })

                    # 分类指标统计
                    category_false_positives[prediction] += 1
                    category_false_negatives[label] += 1

                    # 记录错误匹配
                    mismatches.append({
                        'line_number': line_num,
                        'prediction': prediction,
                        'prediction_reason': analysis['prediction_reason'],
                        'label': label,
                        'label_reason': analysis['label_reason']
                    })

                results.append(analysis)

            except Exception as e:
                print(f"处理第{line_num}行时出错: {str(e)}")
                continue

    # 计算整体指标
    overall_accuracy = correct / total if total > 0 else 0

    # 计算平均相似度指标
    avg_vocab_overlap = total_vocab_overlap / total if total > 0 else 0
    avg_jaccard_sim = total_jaccard_sim / total if total > 0 else 0
    avg_cosine_sim = total_cosine_sim / total if total > 0 else 0

    # 计算精确率、召回率、F1分数
    precision_micro = precision_score(all_true_labels, all_predicted_labels, average='micro', zero_division=0)
    recall_micro = recall_score(all_true_labels, all_predicted_labels, average='micro', zero_division=0)
    f1_micro = f1_score(all_true_labels, all_predicted_labels, average='micro', zero_division=0)

    precision_macro = precision_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
    recall_macro = recall_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)
    f1_macro = f1_score(all_true_labels, all_predicted_labels, average='macro', zero_division=0)

    precision_weighted = precision_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

    # 计算按类别的详细指标
    for category in all_categories:
        cat_total = categories[category]['total'] if category in categories else 0
        cat_correct = categories[category]['correct'] if category in categories else 0
        cat_errors = categories[category]['errors'] if category in categories else 0

        if category not in categories:
            categories[category] = {
                'total': 0, 'correct': 0, 'errors': 0, 'vocab_overlap': 0,
                'jaccard_sim': 0, 'cosine_sim': 0, 'error_details': []
            }

        # 计算每个类别的精确率、召回率、F1
        tp = category_true_positives[category]
        fp = category_false_positives[category]
        fn = category_false_negatives[category]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        categories[category]['accuracy'] = cat_correct / cat_total if cat_total > 0 else 0
        categories[category]['precision'] = precision
        categories[category]['recall'] = recall
        categories[category]['f1_score'] = f1

        # 计算平均相似度指标
        if cat_total > 0:
            categories[category]['avg_vocab_overlap'] = categories[category]['vocab_overlap'] / cat_total
            categories[category]['avg_jaccard_sim'] = categories[category]['jaccard_sim'] / cat_total
            categories[category]['avg_cosine_sim'] = categories[category]['cosine_sim'] / cat_total
        else:
            categories[category]['avg_vocab_overlap'] = 0
            categories[category]['avg_jaccard_sim'] = 0
            categories[category]['avg_cosine_sim'] = 0

    return {
        'total_samples': total,
        'correct_predictions': correct,
        'overall_accuracy': overall_accuracy,
        'overall_accuracy_percentage': overall_accuracy * 100,
        'overall_precision_micro': precision_micro,
        'overall_recall_micro': recall_micro,
        'overall_f1_micro': f1_micro,
        'overall_precision_macro': precision_macro,
        'overall_recall_macro': recall_macro,
        'overall_f1_macro': f1_macro,
        'overall_precision_weighted': precision_weighted,
        'overall_recall_weighted': recall_weighted,
        'overall_f1_weighted': f1_weighted,
        'avg_vocab_overlap': avg_vocab_overlap,
        'avg_jaccard_sim': avg_jaccard_sim,
        'avg_cosine_sim': avg_cosine_sim,
        'category_metrics': dict(categories),
        'mismatches': mismatches,
        'detailed_results': results
    }


def analyze_file_info(filename):
    """分析单个文件的基本信息"""
    info = {
        'filename': filename,
        'file_type': 'K版本' if '_k.' in filename else '标准版本',
        'extension': 'jsonl'
    }

    # 确定模型类别和版本
    if 'Distill_Qwen-14B' in filename:
        info['category'] = 'Distill_Qwen-14B'
        info['model'] = 'Distill Qwen'
        info['version'] = '14B'
    elif 'llama3_8b' in filename:
        info['category'] = 'llama3_8b'
        info['model'] = 'LLaMA3'
        info['version'] = '8B'
    elif 'llama_sft' in filename:
        info['category'] = 'llama_sft'
        info['model'] = 'LLaMA SFT'
        info['version'] = '-'
    elif 'Qwen2.5-7B' in filename:
        info['category'] = 'Qwen2.5-7B'
        info['model'] = 'Qwen2.5'
        info['version'] = '7B'
    elif 'Qwen2.5-32B' in filename:
        info['category'] = 'Qwen2.5-32B'
        info['model'] = 'Qwen2.5'
        info['version'] = '32B'
    elif 'Qwen2.5-0.5B' in filename:
        info['category'] = 'Qwen2.5-0.5B'
        info['model'] = 'Qwen2.5'
        info['version'] = '0.5B'
    elif 'Qwen2.5-1.5B' in filename:
        info['category'] = 'Qwen2.5-1.5B'
        info['model'] = 'Qwen2.5'
        info['version'] = '1.5B'
    elif 'Qwen2.5-3B' in filename:
        info['category'] = 'Qwen2.5-3B'
        info['model'] = 'Qwen2.5'
        info['version'] = '3B'
    elif 'Qwen2.5-14B' in filename:
        info['category'] = 'Qwen2.5-14B'
        info['model'] = 'Qwen2.5'
        info['version'] = '14B'
    elif 'Qwen3-0.6B' in filename:
        info['category'] = 'Qwen3-0.6B'
        info['model'] = 'Qwen3'
        info['version'] = '0.6B'
    elif 'Qwen3-4B' in filename:
        info['category'] = 'Qwen3-4B'
        info['model'] = 'Qwen3'
        info['version'] = '4B'
    elif 'Qwen3-8B' in filename:
        info['category'] = 'Qwen3-8B'
        info['model'] = 'Qwen3'
        info['version'] = '8B'
    elif 'Qwen3-14B' in filename:
        info['category'] = 'Qwen3-14B'
        info['model'] = 'Qwen3'
        info['version'] = '14B'
    elif 'qwen3_sft' in filename:
        info['category'] = 'qwen3_sft'
        info['model'] = 'Qwen3 SFT'
        info['version'] = '-'
    elif 'qwen_sft' in filename:
        info['category'] = 'qwen_sft'
        info['model'] = 'Qwen SFT'
        info['version'] = '-'
    else:
        info['category'] = 'unknown'
        info['model'] = 'unknown'
        info['version'] = 'unknown'

    return info

import glob
def analyze_all_files_predictions(folder_path='.'):
    """分析所有文件的预测结果并导出到Excel"""

    # 动态获取文件列表：所有 .json 和 .jsonl
    patterns = ['*.json', '*.jsonl']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder_path, pattern)))
    files = sorted(files)

    # 存储所有文件的预测结果
    all_results = []
    successful_files = []
    failed_files = []

    print("开始分析所有文件的预测结果...")
    print("=" * 80)

    for i, filename in enumerate(files, 1):
        print(f"[{i:2d}/{len(files)}] 正在分析: {os.path.basename(filename)}")

        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"❌ 文件不存在: {filename}")
            failed_files.append({'filename': filename, 'error': '文件不存在', 'status': 'failed'})
            continue

        try:
            # 分析预测结果
            results = analyze_predictions_file(filename)
            file_info = analyze_file_info(filename)

            # 合并文件信息和预测结果
            combined_result = {
                'filename': filename,
                'model': file_info['model'],
                'version': file_info['version'],
                'file_type': file_info['file_type'],
                'category': file_info['category'],
                'status': 'success',
                **results
            }

            all_results.append(combined_result)
            successful_files.append(filename)

            print(f"✅ 分析完成 - 准确率: {results['overall_accuracy_percentage']:.2f}% "
                  f"(样本数: {results['total_samples']})")

        except Exception as e:
            print(f"❌ 分析失败: {str(e)}")
            failed_files.append({'filename': filename, 'error': str(e), 'status': 'failed'})

    print("\n" + "=" * 80)
    print(f"分析完成! 成功: {len(successful_files)}, 失败: {len(failed_files)}")

    if not all_results:
        print("❌ 没有成功分析的文件，无法生成报告")
        return None, []

    # 创建Excel报告
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    excel_filename = f'所有文件预测结果分析_{current_date}.xlsx'

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:

        # 工作表1：总体概览
        overview_data = []
        for result in all_results:
            overview_data.append([
                result['filename'],
                result['model'],
                result['version'],
                result['file_type'],
                result['total_samples'],
                result['correct_predictions'],
                f"{result['overall_accuracy_percentage']:.2f}%",
                f"{result['overall_precision_micro']:.4f}",
                f"{result['overall_recall_micro']:.4f}",
                f"{result['overall_f1_micro']:.4f}",
                f"{result['avg_vocab_overlap']:.2f}",
                f"{result['avg_jaccard_sim']:.4f}",
                f"{result['avg_cosine_sim']:.4f}"
            ])

        overview_df = pd.DataFrame(overview_data, columns=[
            '文件名', '模型', '版本', '文件类型', '总样本数', '正确预测数', '准确率',
            '微平均精确率', '微平均召回率', '微平均F1', '平均词汇重叠', 'Jaccard相似度', '余弦相似度'
        ])
        overview_df.to_excel(writer, sheet_name='总体概览', index=False)

        # 工作表2：按文件类别指标汇总
        # 创建一个以文件为行，类别指标为列的表格
        file_category_data = []

        # 首先确定所有可能的类别
        all_categories = set()
        for result in all_results:
            all_categories.update(result['category_metrics'].keys())
        all_categories = sorted(list(all_categories))

        for result in all_results:
            row_data = [
                result['filename'],
                result['model'],
                result['file_type']
            ]

            # 为每个类别添加准确率和相似度指标
            for category in all_categories:
                if category in result['category_metrics']:
                    metrics = result['category_metrics'][category]
                    row_data.extend([
                        f"{metrics['accuracy'] * 100:.2f}%",
                        f"{metrics['avg_vocab_overlap']:.2f}",
                        f"{metrics['avg_jaccard_sim']:.4f}",
                        f"{metrics['avg_cosine_sim']:.4f}"
                    ])
                else:
                    # 如果该文件没有这个类别的数据，填入N/A
                    row_data.extend(['N/A', 'N/A', 'N/A', 'N/A'])

            file_category_data.append(row_data)

        # 构建列名
        columns = ['文件名', '模型', '文件类型']
        for category in all_categories:
            columns.extend([
                f'{category}_准确率',
                f'{category}_词汇重叠',
                f'{category}_Jaccard相似度',
                f'{category}_余弦相似度'
            ])

        file_category_df = pd.DataFrame(file_category_data, columns=columns)
        file_category_df.to_excel(writer, sheet_name='按文件类别指标汇总', index=False)

        # 工作表3：类别详细统计
        category_data = []
        for result in all_results:
            filename = result['filename']
            model = result['model']
            file_type = result['file_type']

            for category, metrics in result['category_metrics'].items():
                category_data.append([
                    filename,
                    model,
                    file_type,
                    category,
                    metrics['total'],
                    metrics['correct'],
                    metrics['errors'],
                    f"{metrics['accuracy'] * 100:.2f}%",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1_score']:.4f}",
                    f"{metrics['avg_vocab_overlap']:.2f}",
                    f"{metrics['avg_jaccard_sim']:.4f}",
                    f"{metrics['avg_cosine_sim']:.4f}"
                ])

        category_df = pd.DataFrame(category_data, columns=[
            '文件名', '模型', '文件类型', '类别', '总样本数', '正确预测数', '错误预测数',
            '准确率', '精确率', '召回率', 'F1分数', '平均词汇重叠', 'Jaccard相似度', '余弦相似度'
        ])
        category_df.to_excel(writer, sheet_name='类别详细统计', index=False)

        # 工作表4：类别汇总统计
        # 按类别汇总所有文件的统计
        category_summary = defaultdict(lambda: {
            'total_files': 0, 'total_samples': 0, 'total_correct': 0, 'total_errors': 0,
            'accuracies': [], 'precisions': [], 'recalls': [], 'f1_scores': []
        })

        for result in all_results:
            for category, metrics in result['category_metrics'].items():
                category_summary[category]['total_files'] += 1
                category_summary[category]['total_samples'] += metrics['total']
                category_summary[category]['total_correct'] += metrics['correct']
                category_summary[category]['total_errors'] += metrics['errors']
                category_summary[category]['accuracies'].append(metrics['accuracy'])
                category_summary[category]['precisions'].append(metrics['precision'])
                category_summary[category]['recalls'].append(metrics['recall'])
                category_summary[category]['f1_scores'].append(metrics['f1_score'])

        category_summary_data = []
        for category, stats in category_summary.items():
            overall_accuracy = (stats['total_correct'] / stats['total_samples'] * 100) if stats[
                                                                                              'total_samples'] > 0 else 0
            avg_accuracy = np.mean(stats['accuracies']) * 100
            avg_precision = np.mean(stats['precisions'])
            avg_recall = np.mean(stats['recalls'])
            avg_f1 = np.mean(stats['f1_scores'])

            category_summary_data.append([
                category,
                stats['total_files'],
                stats['total_samples'],
                stats['total_correct'],
                stats['total_errors'],
                f"{overall_accuracy:.2f}%",
                f"{avg_accuracy:.2f}%",
                f"{avg_precision:.4f}",
                f"{avg_recall:.4f}",
                f"{avg_f1:.4f}"
            ])

        category_summary_df = pd.DataFrame(category_summary_data, columns=[
            '类别', '涉及文件数', '总样本数', '总正确数', '总错误数',
            '整体准确率', '平均准确率', '平均精确率', '平均召回率', '平均F1分数'
        ])
        category_summary_df.to_excel(writer, sheet_name='类别汇总统计', index=False)

        # 工作表5：详细评估指标
        detailed_data = []
        for result in all_results:
            detailed_data.append([
                result['filename'],
                result['model'],
                result['total_samples'],
                f"{result['overall_accuracy_percentage']:.2f}%",
                f"{result['overall_precision_micro']:.4f}",
                f"{result['overall_recall_micro']:.4f}",
                f"{result['overall_f1_micro']:.4f}",
                f"{result['overall_precision_macro']:.4f}",
                f"{result['overall_recall_macro']:.4f}",
                f"{result['overall_f1_macro']:.4f}",
                f"{result['overall_precision_weighted']:.4f}",
                f"{result['overall_recall_weighted']:.4f}",
                f"{result['overall_f1_weighted']:.4f}"
            ])

        detailed_df = pd.DataFrame(detailed_data, columns=[
            '文件名', '模型', '总样本数', '准确率',
            '微平均精确率', '微平均召回率', '微平均F1',
            '宏平均精确率', '宏平均召回率', '宏平均F1',
            '加权平均精确率', '加权平均召回率', '加权平均F1'
        ])
        detailed_df.to_excel(writer, sheet_name='详细评估指标', index=False)

        # 工作表6：按模型类型汇总
        model_summary = defaultdict(lambda: {
            'files': [], 'total_samples': 0, 'total_correct': 0,
            'accuracies': [], 'f1_scores': []
        })

        for result in all_results:
            model = result['model']
            model_summary[model]['files'].append(result['filename'])
            model_summary[model]['total_samples'] += result['total_samples']
            model_summary[model]['total_correct'] += result['correct_predictions']
            model_summary[model]['accuracies'].append(result['overall_accuracy_percentage'])
            model_summary[model]['f1_scores'].append(result['overall_f1_micro'])

        model_data = []
        for model, stats in model_summary.items():
            avg_accuracy = np.mean(stats['accuracies'])
            avg_f1 = np.mean(stats['f1_scores'])
            overall_accuracy = (stats['total_correct'] / stats['total_samples'] * 100) if stats[
                                                                                              'total_samples'] > 0 else 0

            model_data.append([
                model,
                len(stats['files']),
                stats['total_samples'],
                stats['total_correct'],
                f"{overall_accuracy:.2f}%",
                f"{avg_accuracy:.2f}%",
                f"{avg_f1:.4f}",
                '; '.join(stats['files'])
            ])

        model_df = pd.DataFrame(model_data, columns=[
            '模型', '文件数', '总样本数', '总正确数', '整体准确率', '平均准确率', '平均F1', '文件列表'
        ])
        model_df.to_excel(writer, sheet_name='按模型汇总', index=False)

        # 工作表7：文件类型对比 (标准版本 vs K版本)
        type_comparison = defaultdict(lambda: {
            'files': [], 'total_samples': 0, 'total_correct': 0,
            'accuracies': [], 'f1_scores': []
        })

        for result in all_results:
            file_type = result['file_type']
            type_comparison[file_type]['files'].append(result['filename'])
            type_comparison[file_type]['total_samples'] += result['total_samples']
            type_comparison[file_type]['total_correct'] += result['correct_predictions']
            type_comparison[file_type]['accuracies'].append(result['overall_accuracy_percentage'])
            type_comparison[file_type]['f1_scores'].append(result['overall_f1_micro'])

        type_data = []
        for file_type, stats in type_comparison.items():
            avg_accuracy = np.mean(stats['accuracies'])
            avg_f1 = np.mean(stats['f1_scores'])
            overall_accuracy = (stats['total_correct'] / stats['total_samples'] * 100) if stats[
                                                                                              'total_samples'] > 0 else 0

            type_data.append([
                file_type,
                len(stats['files']),
                stats['total_samples'],
                stats['total_correct'],
                f"{overall_accuracy:.2f}%",
                f"{avg_accuracy:.2f}%",
                f"{avg_f1:.4f}"
            ])

        type_df = pd.DataFrame(type_data, columns=[
            '文件类型', '文件数', '总样本数', '总正确数', '整体准确率', '平均准确率', '平均F1'
        ])
        type_df.to_excel(writer, sheet_name='文件类型对比', index=False)

        # 工作表8：错误匹配详情（如果有的话）
        if any('mismatches' in result and result['mismatches'] for result in all_results):
            mismatch_data = []
            for result in all_results:
                if 'mismatches' in result and result['mismatches']:
                    for mismatch in result['mismatches']:
                        mismatch_data.append([
                            result['filename'],
                            mismatch['line_number'],
                            mismatch['prediction'],
                            mismatch['label'],
                            mismatch.get('prediction_reason', '')[:100] + "..." if len(
                                mismatch.get('prediction_reason', '')) > 100 else mismatch.get('prediction_reason', ''),
                            mismatch.get('label_reason', '')[:100] + "..." if len(
                                mismatch.get('label_reason', '')) > 100 else mismatch.get('label_reason', '')
                        ])

            if mismatch_data:
                mismatch_df = pd.DataFrame(mismatch_data, columns=[
                    '文件名', '行号', '预测结果', '真实标签', '预测理由', '标签理由'
                ])
                mismatch_df.to_excel(writer, sheet_name='错误匹配详情', index=False)

        # 工作表9：失败文件列表
        if failed_files:
            failed_df = pd.DataFrame(failed_files)
            failed_df.to_excel(writer, sheet_name='失败文件', index=False)

    # 打印汇总统计
    print("\n" + "🏆 最佳表现文件 (按准确率排序)")
    print("-" * 80)
    sorted_results = sorted(all_results, key=lambda x: x['overall_accuracy_percentage'], reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"{i:2d}. {result['filename']:<35} "
              f"准确率: {result['overall_accuracy_percentage']:6.2f}% "
              f"F1: {result['overall_f1_micro']:.4f} "
              f"样本: {result['total_samples']:4d}")

    print(f"\n✅ 详细Excel报告已保存: {excel_filename}")

    return excel_filename, all_results


def print_overall_summary(all_results):
    """打印整体统计摘要"""
    if not all_results:
        print("\n❌ 没有成功分析任何文件")
        return

    print("\n" + "=" * 80)
    print("📈 整体统计摘要")
    print("=" * 80)

    # 计算整体统计
    total_files_analyzed = len(all_results)
    total_samples = sum(r['total_samples'] for r in all_results)
    total_correct = sum(r['correct_predictions'] for r in all_results)
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0

    avg_accuracy = np.mean([r['overall_accuracy_percentage'] for r in all_results])
    avg_f1 = np.mean([r['overall_f1_micro'] for r in all_results])

    print(f"成功分析文件数: {total_files_analyzed}")
    print(f"总样本数: {total_samples:,}")
    print(f"总正确预测数: {total_correct:,}")
    print(f"整体准确率: {overall_accuracy:.2f}%")
    print(f"平均准确率: {avg_accuracy:.2f}%")
    print(f"平均F1分数: {avg_f1:.4f}")

    # 按类别统计
    print("\n📊 按类别统计:")
    print("-" * 60)
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0})

    for result in all_results:
        for category, metrics in result['category_metrics'].items():
            category_stats[category]['total'] += metrics['total']
            category_stats[category]['correct'] += metrics['correct']
            category_stats[category]['errors'] += metrics['errors']

    for category, stats in sorted(category_stats.items()):
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{category:<8}: 总样本 {stats['total']:5d}, 正确 {stats['correct']:5d}, "
              f"错误 {stats['errors']:4d}, 准确率 {accuracy:6.2f}%")

    # 按模型类型统计
    print("\n📊 按模型类型统计:")
    print("-" * 50)
    model_stats = defaultdict(lambda: {'files': 0, 'samples': 0, 'correct': 0, 'accuracies': []})

    for result in all_results:
        model = result['model']
        model_stats[model]['files'] += 1
        model_stats[model]['samples'] += result['total_samples']
        model_stats[model]['correct'] += result['correct_predictions']
        model_stats[model]['accuracies'].append(result['overall_accuracy_percentage'])

    for model, stats in sorted(model_stats.items()):
        overall_acc = (stats['correct'] / stats['samples'] * 100) if stats['samples'] > 0 else 0
        avg_acc = np.mean(stats['accuracies'])
        print(f"{model:<15}: {stats['files']} 文件, 整体准确率: {overall_acc:6.2f}%, 平均准确率: {avg_acc:6.2f}%")

    # 按文件类型对比 (标准版本 vs K版本)
    print("\n🔄 标准版本 vs K版本对比:")
    print("-" * 40)
    type_stats = defaultdict(lambda: {'files': 0, 'samples': 0, 'correct': 0, 'accuracies': []})

    for result in all_results:
        file_type = result['file_type']
        type_stats[file_type]['files'] += 1
        type_stats[file_type]['samples'] += result['total_samples']
        type_stats[file_type]['correct'] += result['correct_predictions']
        type_stats[file_type]['accuracies'].append(result['overall_accuracy_percentage'])

    for file_type, stats in sorted(type_stats.items()):
        overall_acc = (stats['correct'] / stats['samples'] * 100) if stats['samples'] > 0 else 0
        avg_acc = np.mean(stats['accuracies'])
        print(f"{file_type:<10}: {stats['files']} 文件, 整体准确率: {overall_acc:6.2f}%, 平均准确率: {avg_acc:6.2f}%")


def save_category_metrics_csv(all_results):
    """保存每个文件按类别的指标到CSV文件"""
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_filename = f'按文件类别指标详情_{current_date}.csv'

    # 首先确定所有可能的类别
    all_categories = set()
    for result in all_results:
        all_categories.update(result['category_metrics'].keys())
    all_categories = sorted(list(all_categories))

    csv_data = []
    for result in all_results:
        for category in all_categories:
            if category in result['category_metrics']:
                metrics = result['category_metrics'][category]
                csv_data.append([
                    result['filename'],
                    result['model'],
                    result['version'],
                    result['file_type'],
                    category,
                    metrics['total'],
                    metrics['correct'],
                    metrics['errors'],
                    f"{metrics['accuracy'] * 100:.2f}",
                    f"{metrics['avg_vocab_overlap']:.2f}",
                    f"{metrics['avg_jaccard_sim']:.4f}",
                    f"{metrics['avg_cosine_sim']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1_score']:.4f}"
                ])

    # 保存为CSV
    csv_df = pd.DataFrame(csv_data, columns=[
        '文件名', '模型', '版本', '文件类型', '类别',
        '总样本数', '正确预测数', '错误预测数', '准确率(%)',
        '平均词汇重叠', 'Jaccard相似度', '余弦相似度',
        '精确率', '召回率', 'F1分数'
    ])

    csv_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    print(f"✅ 按类别指标详情已保存至CSV文件: {csv_filename}")
    return csv_filename


# 使用示例
if __name__ == "__main__":
    print("🚀 开始批量分析所有文件的预测结果...")

    # 分析所有文件的预测结果
    excel_file, all_results = analyze_all_files_predictions()

    # 保存按类别指标的CSV文件
    if all_results:
        csv_file = save_category_metrics_csv(all_results)

    # 打印整体统计摘要
    print_overall_summary(all_results)

    print("\n" + "=" * 80)
    print("✅ 分析完成！详细结果请查看生成的Excel和CSV文件。")
    print("=" * 80)
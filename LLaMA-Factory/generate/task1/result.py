import json
import os
import pandas as pd
import re

folder_path = ""
results = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jsonl") or filename.endswith(".json") and (filename.startswith("sft") or filename.startswith("nosft")):
        file_path = os.path.join(folder_path, filename)
        total = 0
        correct = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                try:
                    predict = data.get("predict","none").strip()
                except:
                    predict = ""
                try:
                   label = data.get("label").strip()
                except:
                   label = data.get("output").strip()
                if "assistantfinal" in predict:
                    predict = predict.split("assistantfinal")[-1].strip()
                if "assistantfinal" in label:
                    label = label.split("analysisassistantfinal")[-1].strip()

                total += 1  # ✅ 每条样本都计入总数

                if "<think>" in predict:
                    parts = predict.split("</think>")
                    try:
                        predict = parts[1].strip()
                    except:
                        predict=""

                if "<think>" in label:
                    parts = label.split("</think>")
                    label = parts[1].strip()

                if "家庭成员关系" in filename:
                    pred_list = re.findall(r"'(.*?)'", predict)
                    label_list = re.findall(r"'(.*?)'", label)
                    if bool(set(pred_list) & set(label_list)):
                        correct += 1

                elif "家里人口数" in filename:
                    try:
                        pred_list = re.findall(r'\d+', predict)
                        if len(pred_list)==1:
                            if label == pred_list[0]:
                                correct += 1
                    except:
                        pass

                else:
                    if label.startswith("[") and label.endswith("]"):
                        if "主要支出" in filename:
                            match_label = re.findall(r"'(.*?)'", label)
                            if match_label:
                                label = match_label[0]
                                label = re.sub(r"（.*?）|\(.*?\)", "", label).strip()
                            match_predict = re.findall(r"'(.*?)'", predict)
                            if match_predict:
                                predict = match_predict[0]
                                predict = re.sub(r"（.*?）|\(.*?\)", "", predict).strip()
                        if "收入" in filename:
                            label = label.strip("[]").strip("'\"")
                            predict = predict.strip("[]").strip("'\"")

                    if label.strip() == predict.strip():
                        correct += 1

        acc = correct / total if total > 0 else 0
        print({"filename": filename, "accuracy": acc, "total": total, "correct": correct})
        results.append({"filename": filename, "accuracy": acc, "total": total, "correct": correct})

df = pd.DataFrame(results)
df.to_excel("accuracy_results.xlsx", index=False)

grouped_results = {}

for result in results:
    filename = result["filename"]
    try:
        match = re.search(r"back_(.*?)\.jsonl$", filename)
    except:
        match = re.search(r"test_(.*?)\.jsonl$", filename)
    if match:
        topic = match.group(1)
        if topic not in grouped_results:
            grouped_results[topic] = []
        grouped_results[topic].append(result)

output_dir = "result"
os.makedirs(output_dir, exist_ok=True)

for topic, group in grouped_results.items():
    df_topic = pd.DataFrame(group)
    df_topic.to_excel(os.path.join(output_dir, f"{topic}.xlsx"), index=False)

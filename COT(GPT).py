API_KEY_OPENAI = ""  
RUN_COUNT = 1000                # 运行前1000条数据

import requests
import os
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI
import re
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

client = OpenAI(
    api_key=API_KEY_OPENAI,
    base_url="https://api.openai.com/v1",
)

def gpt_35_turbo(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "你是SQL专家，请一步步推理，最后输出可执行的SQL语句。"},
                      {"role": "user", "content": prompt}],
            temperature=0.0
        )
        return completion.choices[0].message.content.strip()
    except:
        return ""

MODEL_LIST = [gpt_35_turbo]
def extract_sql_from_response(response_text):
    """
    从 CoT 回答中提取 SQL 语句。
    优先提取 SQL 代码块，若无则尝试提取 SELECT/INSERT/UPDATE/DELETE 开头的行。
    """
    if not response_text:
        return ""
    patterns = [
        r'```sql\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(SELECT\s+.*?;?)',
        r'(INSERT\s+.*?;?)',
        r'(UPDATE\s+.*?;?)',
        r'(DELETE\s+.*?;?)'
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1).strip()
            if sql:
                return sql
    return response_text.strip()

def sql_similarity(pred_sql, true_sql):
    """返回标准化后的字符串相似度（0~1）"""
    pred_clean = re.sub(r'\s+', ' ', pred_sql.strip().upper())
    true_clean = re.sub(r'\s+', ' ', true_sql.strip().upper())
    return SequenceMatcher(None, pred_clean, true_clean).ratio()

def exact_match(pred_sql, true_sql):
    """严格匹配：标准化后字符串完全相同"""
    pred_clean = re.sub(r'\s+', ' ', pred_sql.strip().upper())
    true_clean = re.sub(r'\s+', ' ', true_sql.strip().upper())
    return pred_clean == true_clean

DATASET_NAME = "toolbench"
SPLIT = "train"
FIELD_MAP = {"question": "question", "answer": "SQL"}

def get_parquet_path(dataset_name, split):
    split_file_map = {
        "train": "train-00000-of-00001",
        "test": "test-00000-of-00001",
        "validation": "validation-00000-of-00001"
    }
    base_name = split_file_map[split]
    path_candidates = [
        Path("./data") / dataset_name / f"{base_name}.parquet",
        Path("./data") / dataset_name / f"{base_name} (1).parquet",
    ]
    for p in path_candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到 {dataset_name}/{split} 的 parquet 文件")

def load_parquet_data(dataset_name, split, field_map, limit=RUN_COUNT):
    file_path = get_parquet_path(dataset_name, split)
    df = pd.read_parquet(file_path)
    stream_data = []
    for _, row in df.iterrows():
        item = {
            "question": str(row[field_map["question"]]),
            "answer": str(row[field_map["answer"]])
        }
        stream_data.append(item)
        if len(stream_data) >= limit:
            break
    return stream_data

stream_data = load_parquet_data(DATASET_NAME, SPLIT, FIELD_MAP)
print(f"成功加载 {len(stream_data)} 条数据\n")

all_extracted_sqls = []    # 提取出的SQL
all_trues = []             # 真实SQL字符串
all_model_names = []       # 使用的模型名称
all_valid = []             # 是否有效输出

for idx, item in enumerate(stream_data):
    model = MODEL_LIST[idx % len(MODEL_LIST)]
    model_name = model.__name__
    q = item["question"]
    true_sql = item["answer"]

    prompt = f"""请一步步思考如何回答用户的问题，最后输出对应的 SQL 语句（用```sql...```包裹）。
用户问题：{q}
逐步推理："""

    response = model(prompt)

    # 提取SQL
    extracted_sql = extract_sql_from_response(response)

    # 判断是否有效输出
    if not extracted_sql or len(extracted_sql) < 3:
        valid = False
        extracted_sql = ""
    else:
        valid = True

    all_extracted_sqls.append(extracted_sql)
    all_trues.append(true_sql)
    all_model_names.append(model_name)
    all_valid.append(valid)

    print(f"===== 第{idx+1}题 | 模型：{model_name} =====")
    print(f"问题：{q[:80]}...")
    print(f"模型原始回答摘要：{response[:150]}...")
    print(f"提取的SQL：{extracted_sql[:100] if extracted_sql else '无效输出'}...")
    print(f"正确答案：{true_sql[:100]}...")
    print(f"有效输出：{valid}\n")

similarity_correct = []
exact_match_correct = []
for i, (pred, true) in enumerate(zip(all_extracted_sqls, all_trues)):
    if not all_valid[i] or pred == "":
        similarity_correct.append(False)
        exact_match_correct.append(False)
    else:
        sim = sql_similarity(pred, true)
        similarity_correct.append(sim > 0.7)
        exact_match_correct.append(exact_match(pred, true))

total_samples = len(stream_data)
valid_samples = sum(all_valid)

acc_similarity = np.mean(similarity_correct)
acc_exact = np.mean(exact_match_correct)
f1 = f1_score([1]*len(similarity_correct), similarity_correct, average='binary', zero_division=0)
mcc = matthews_corrcoef([1]*len(similarity_correct), similarity_correct)

model_stats = {}
for model_name in set(all_model_names):
    indices = [i for i, name in enumerate(all_model_names) if name == model_name]
    if not indices:
        continue
    sim_corr = [similarity_correct[i] for i in indices]
    exact_corr = [exact_match_correct[i] for i in indices]
    valid_in_model = sum(all_valid[i] for i in indices)
    model_stats[model_name] = {
        "count": len(indices),
        "valid": valid_in_model,
        "acc_sim": np.mean(sim_corr),
        "acc_exact": np.mean(exact_corr),
        "f1": f1_score([1]*len(sim_corr), sim_corr, zero_division=0),
        "mcc": matthews_corrcoef([1]*len(sim_corr), sim_corr)
    }

print("="*80)
print("Chain-of-Thought (CoT) 评估结果（基于相似度阈值 0.7）")
print("="*80)
print(f"总样本数：{total_samples}")
print(f"有效输出数：{valid_samples} ({valid_samples/total_samples*100:.1f}%)")
print(f"无效输出数：{total_samples - valid_samples}")
print(f"\n整体指标（基于相似度>0.7的正确性）：")
print(f" ACC（相似度准确率）：{acc_similarity:.4f}")
print(f" F1 Score：{f1:.4f}")
print(f" Matthews Correlation Coefficient (MCC)：{mcc:.4f}")
print(f"\n整体严格匹配指标：")
print(f" Exact Match 准确率：{acc_exact:.4f}")
print(f"\n各模型详细指标（基于相似度>0.7）：")
for model_name in model_stats:
    s = model_stats[model_name]
    print(f"\n   {model_name}:")
    print(f" 答题数：{s['count']} | 有效输出：{s['valid']} ({s['valid']/s['count']*100:.1f}%)")
    print(f" ACC (相似度)：{s['acc_sim']:.4f}")
    print(f" Exact Match：{s['acc_exact']:.4f}")
    print(f" F1：{s['f1']:.4f}")
    print(f" MCC：{s['mcc']:.4f}")

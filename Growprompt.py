API_KEY_QWEN = ""
API_KEY_ZHIPU = ""
RUN_COUNT = 1000                # 运行前1000条数据
GROW_WINDOW_SIZE = 16           # GrowPrompt 滑动窗口大小（论文默认k=16）

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
    api_key="",
    base_url="https://api.moonshot.cn/v1",
)
def kimi(prompt):
    try:
        completion = client.chat.completions.create(
            model="kimi-k2.6",
            messages=[{"role": "system", "content": "你是SQL专家，只输出可执行的SQL语句，不要解释。"},
                      {"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except:
        return ""

api_key_doubao = os.getenv('')
client_doubao = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key_doubao,
)
def doubao(prompt):
    try:
        response = client_doubao.responses.create(
            model="doubao-seed-2-0-lite-260215",
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        )
        return response.output.choices[0].message.content.strip()
    except:
        return ""

def qwen(prompt):
    try:
        headers = {"Authorization": f"Bearer {API_KEY_QWEN}"}
        data = {
            "model": "qwen-turbo",
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": {"temperature": 0.0}
        }
        resp = requests.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                             headers=headers, json=data)
        return resp.json()["output"]["text"].strip()
    except:
        return ""

def zhipu(prompt):
    try:
        headers = {"Authorization": f"Bearer {API_KEY_ZHIPU}"}
        data = {
            "model": "glm-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        resp = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                             headers=headers, json=data)
        return resp.json()["choices"][0]["message"]["content"].strip()
    except:
        return ""

MODEL_LIST = [kimi, qwen, zhipu, doubao]
MODEL_NAMES = [m.__name__ for m in MODEL_LIST]

def sql_similarity(pred_sql, true_sql):
    pred_clean = re.sub(r'\s+', ' ', pred_sql.strip().upper())
    true_clean = re.sub(r'\s+', ' ', true_sql.strip().upper())
    return SequenceMatcher(None, pred_clean, true_clean).ratio()

def exact_match(pred_sql, true_sql):
    pred_clean = re.sub(r'\s+', ' ', pred_sql.strip().upper())
    true_clean = re.sub(r'\s+', ' ', true_sql.strip().upper())
    return pred_clean == true_clean

def build_grow_prompt(question, window_examples):
    """
    构建 GrowPrompt 提示词：拼接滑动窗口内的历史样本+文字化反馈
    window_examples: [(question, pred_sql, is_correct), ...]
    """
    example_text = ""
    for idx, (ex_q, ex_sql, correct) in enumerate(window_examples):
        feedback = "回答正确" if correct else "回答不正确"
        example_text += f"""历史问题{idx+1}：{ex_q}
模型输出SQL：{ex_sql}
反馈：{feedback}

"""
    prompt = f"""以下是历史问答记录及正确性反馈，请参考后回答新问题。
{example_text}
现在处理新问题，只输出SQL语句，不要解释。
用户问题：{question}
SQL语句："""
    return prompt


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

#GrowPrompt
grow_window = []  # 存储 (question, pred_sql, is_correct)

all_preds = []
all_trues = []
all_model_names = []
all_valid = []

for idx, item in enumerate(stream_data):
    # 轮询选择模型
    agent_idx = idx % len(MODEL_LIST)
    model = MODEL_LIST[agent_idx]
    model_name = MODEL_NAMES[agent_idx]
    
    q = item["question"]
    true_sql = item["answer"]
    
    # 用当前滑动窗口构建 prompt
    prompt = build_grow_prompt(q, grow_window)
    
    # 模型预测
    pred_sql = model(prompt)
    
    # 有效性判断
    if not pred_sql or len(pred_sql) < 3 or "无法回答" in pred_sql or "Unable" in pred_sql:
        valid = False
        pred_sql = ""
    else:
        valid = True
    
    # 计算正确性
    if valid:
        sim = sql_similarity(pred_sql, true_sql)
        is_correct_sim = sim > 0.7
        is_correct_exact = exact_match(pred_sql, true_sql)
    else:
        is_correct_sim = False
        is_correct_exact = False
    
    # 更新 GrowPrompt 
    grow_window.append((q, pred_sql, is_correct_sim))
    if len(grow_window) > GROW_WINDOW_SIZE:
        grow_window.pop(0)  # 移除最旧的样本
    
    # 记录结果
    all_preds.append(pred_sql)
    all_trues.append(true_sql)
    all_model_names.append(model_name)
    all_valid.append(valid)
    
    print(f"===== 第{idx+1}题 | 模型：{model_name} | 窗口大小：{len(grow_window)} =====")
    print(f"问题：{q[:80]}...")
    print(f"模型回答：{pred_sql[:100] if pred_sql else '无效输出'}...")
    print(f"正确答案：{true_sql[:100]}...")
    print(f"有效输出：{valid} | 相似度正确：{is_correct_sim}\n")

# ===================== 指标计算 =====================
similarity_correct = []
exact_match_correct = []
for i, (pred, true) in enumerate(zip(all_preds, all_trues)):
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

# 各模型细分指标
model_stats = {}
for name in MODEL_NAMES:
    indices = [i for i, n in enumerate(all_model_names) if n == name]
    if not indices:
        continue
    sim_corr = [similarity_correct[i] for i in indices]
    exact_corr = [exact_match_correct[i] for i in indices]
    valid_in_model = sum(all_valid[i] for i in indices)
    model_stats[name] = {
        "count": len(indices),
        "valid": valid_in_model,
        "acc_sim": np.mean(sim_corr),
        "acc_exact": np.mean(exact_corr),
        "f1": f1_score([1]*len(sim_corr), sim_corr, zero_division=0),
        "mcc": matthews_corrcoef([1]*len(sim_corr), sim_corr)
    }

print("="*80)
print("GrowPrompt 评估结果（数据集：toolbench | 相似度阈值 0.7 | 窗口大小 16）")
print("="*80)
print(f"总样本数：{total_samples}")
print(f"有效输出数：{valid_samples} ({valid_samples/total_samples*100:.1f}%)")
print(f"无效输出数：{total_samples - valid_samples}")

print(f"\n整体指标（基于相似度>0.7的正确性）：")
print(f"ACC（相似度准确率）：{acc_similarity:.4f}")
print(f"F1 Score：{f1:.4f}")
print(f"Matthews Correlation Coefficient (MCC)：{mcc:.4f}")

print(f"\n整体严格匹配指标：")
print(f"Exact Match 准确率：{acc_exact:.4f}")

print(f"\n各模型详细指标（基于相似度>0.7）：")
for name in model_stats:
    s = model_stats[name]
    print(f"\n   {name}:")
    print(f" 答题数：{s['count']} | 有效输出：{s['valid']} ({s['valid']/s['count']*100:.1f}%)")
    print(f" ACC (相似度)：{s['acc_sim']:.4f}")
    print(f" Exact Match：{s['acc_exact']:.4f}")
    print(f" F1：{s['f1']:.4f}")
    print(f" MCC：{s['mcc']:.4f}")

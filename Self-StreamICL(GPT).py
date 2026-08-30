API_KEY_OPENAI = ""  
MAX_MEMORY = 16
MEMORY_EVOLVE = True          # 记忆进化开关
MODEL_EVALUATE = True         # 模型动态评分开关
ADAPTIVE_PROMPT = True        # 自适应提示词
RUN_COUNT = 1000               # 评测数据量，可根据需求调整
FINE_GRAINED_DIAGNOSIS = False # 关闭细粒度诊断，加速评测
MSU_PSK_ENABLED = False
P_NUM_VARIANTS = 2
TARGET_MODELS = ["gpt-3.5-turbo"]
TARGET_DATASETS = ["DS-1000", "DDXPlus", "ToolBench"]
TARGET_METRICS = ["Exact Match", "F1 Score", "MCC"]

import requests
import os
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher
import re
import copy
import random
from typing import List, Dict, Tuple

def gpt_35_turbo(prompt, system_prompt=""):
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=API_KEY_OPENAI,
            base_url="https://api.openai.com/v1",
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT-3.5-turbo模型调用失败：{str(e)}")
        return ""

MODEL_FUNC_DICT = {
    "gpt-3.5-turbo": gpt_35_turbo
}

# 模型显示名称映射
MODEL_DISPLAY_NAME = {
    "gpt-3.5-turbo": "GPT-3.5-turbo"
}

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def memory_evolve(memory_list, new_item):
    for item in memory_list:
        if similarity(item["q"], new_item["q"]) > 0.9:
            return False
    memory_list.append(new_item)
    memory_list.sort(key=lambda x: similarity(x["q"], new_item["q"]), reverse=True)
    if len(memory_list) > MAX_MEMORY:
        memory_list.pop(-1)
    return True

# 评测指标计算函数
def standardize_text(text: str, task_type: str) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if task_type == "sql":
        text = text.upper()
        text = re.sub(r'\s+', ' ', text)
        text = text.rstrip(';')
    return text

def calculate_exact_match(y_true: List[str], y_pred: List[str], task_type: str = "sql") -> float:
    """计算Exact Match"""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    match_count = 0
    for true, pred in zip(y_true, y_pred):
        true_std = standardize_text(true, task_type)
        pred_std = standardize_text(pred, task_type)
        if true_std == pred_std:
            match_count += 1
    return round(match_count / len(y_true), 4)

def calculate_f1_score(y_true: List[str], y_pred: List[str], task_type: str = "sql") -> float:
    """计算F1 Score"""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    
    y_true_std = [standardize_text(t, task_type) for t in y_true]
    y_pred_std = [standardize_text(p, task_type) for p in y_pred]
    
    unique_labels = list(set(y_true_std + y_pred_std))
    f1_list = []
    
    for label in unique_labels:
        tp = sum(1 for t, p in zip(y_true_std, y_pred_std) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true_std, y_pred_std) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true_std, y_pred_std) if t == label and p != label)
        
        if tp + fp + fn == 0:
            f1_list.append(0.0)
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_list.append(f1)
    
    return round(np.mean(f1_list) if f1_list else 0.0, 4)

def calculate_mcc(y_true: List[str], y_pred: List[str], task_type: str = "sql") -> float:
    """计算Matthews Correlation Coefficient（MCC）"""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    
    y_true_std = [standardize_text(t, task_type) for t in y_true]
    y_pred_std = [standardize_text(p, task_type) for p in y_pred]
    
    unique_labels = list(set(y_true_std + y_pred_std))
    k = len(unique_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    conf_matrix = np.zeros((k, k), dtype=int)
    for true, pred in zip(y_true_std, y_pred_std):
        true_idx = label_to_idx[true]
        pred_idx = label_to_idx[pred]
        conf_matrix[true_idx, pred_idx] += 1
    
    s = np.sum(conf_matrix)
    c = np.trace(conf_matrix)
    p = np.sum(conf_matrix, axis=0)
    t = np.sum(conf_matrix, axis=1)
    
    numerator = c * s - np.sum(p * t)
    denominator = np.sqrt((s**2 - np.sum(p**2)) * (s**2 - np.sum(t**2)))
    
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)

DATASET_CONFIG = {
    "DS-1000": {
        "task_type": "sql",
        "field_map": {"question": "question", "answer": "answer"},
        "split": "test"
    },
    "DDXPlus": {
        "task_type": "分类",
        "field_map": {"question": "question", "answer": "diagnosis"},
        "split": "test"
    },
    "ToolBench": {
        "task_type": "sql",
        "field_map": {"question": "question", "answer": "SQL"},
        "split": "test"
    }
}

def load_dataset(dataset_name: str) -> List[Dict]:
    config = DATASET_CONFIG[dataset_name]
    split = config["split"]
    field_map = config["field_map"]
    
    split_file_map = {
        "train": "train-00000-of-00001",
        "test": "test-00000-of-00001",
        "validation": "validation-00000-of-00001"
    }
    base_name = split_file_map[split]
    path_candidates = [
        Path(r"C:\Users\liuxi\Desktop\streambench\data\ds_1000\test-00000-of-00001.parquet"),
        Path(r"C:\Users\liuxi\Desktop\streambench\data\ddxplus\test-00000-of-00001 (1).parquet"),
        Path(r"C:\Users\liuxi\Desktop\streambench\data\toolbench\test-00000-of-00001 (1).parquet")
    ]
    
    dataset_path_map = {
        "DS-1000": path_candidates[0],
        "DDXPlus": path_candidates[1],
        "ToolBench": path_candidates[2]
    }
    target_path = dataset_path_map.get(dataset_name)
    
    if target_path and target_path.exists():
        df = pd.read_parquet(target_path)
        stream_data = []
        for _, row in df.iterrows():
            item = {
                "question": str(row[field_map["question"]]),
                "answer": str(row[field_map["answer"]])
            }
            stream_data.append(item)
        return stream_data[:RUN_COUNT]
    
    print(f"提示：未找到{dataset_name}数据集，使用演示数据运行")
    if dataset_name == "DDXPlus":
        return [
            {"question": "患者发热伴咳嗽，诊断是什么？", "answer": "上呼吸道感染"},
            {"question": "患者腹痛腹泻，诊断是什么？", "answer": "急性肠胃炎"},
        ]
    else:
        return [
            {"question": "查询用户表的所有数据", "answer": "SELECT * FROM users"},
            {"question": "查询年龄大于18的用户", "answer": "SELECT * FROM users WHERE age > 18"},
            {"question": "统计用户总数", "answer": "SELECT COUNT(*) FROM users"},
            {"question": "查询订单表的总金额", "answer": "SELECT SUM(amount) FROM orders"},
            {"question": "查询2023年的订单", "answer": "SELECT * FROM orders WHERE year = 2023"},
        ]

def evaluate_model(model_name: str, dataset_name: str) -> Dict:"
    print(f"\n===== 评测：{MODEL_DISPLAY_NAME[model_name]} @ {dataset_name} =====")
    
    dataset = load_dataset(dataset_name)
    task_type = DATASET_CONFIG[dataset_name]["task_type"]
    model_func = MODEL_FUNC_DICT[model_name]
    y_true = []
    y_pred = []
    shared_memory = []
    
    for idx, item in enumerate(dataset):
        q = item["question"]
        true_a = item["answer"]
        y_true.append(true_a)
        
        # 子任务分解
        step1_prompt = f"""
        任务：将用户问题拆解为SQL生成所需的子步骤：
        用户问题：{q}
        请输出：
        1. 需要查询的表名
        2. 过滤条件
        3. 需要的字段
        4. 是否需要JOIN
        """
        decomposition = gpt_35_turbo(step1_prompt)
        
        # 构造带记忆的提示
        memory_prompt = ""
        for m in shared_memory[-MAX_MEMORY:]:
            memory_prompt += f"问题：{m['q']}\nSQL：{m['a']}\n"
        prompt = f"""
        你是SQL专家，根据用户问题生成正确的SQL语句。
        参考示例：
        {memory_prompt}
        用户问题：{q}
        输出要求：只输出SQL语句，不要解释。
        """
        
        pred_a = model_func(prompt, system_prompt="你是SQL专家，只输出可执行的SQL语句，不要解释。")
        y_pred.append(pred_a)
        
        # 记忆进化
        is_correct = similarity(
            standardize_text(pred_a, "sql"),
            standardize_text(true_a, "sql")
        ) > 0.7
        if MEMORY_EVOLVE and is_correct:
            memory_evolve(shared_memory, {"q": q, "a": pred_a})
        
        if (idx + 1) % 10 == 0:
            print(f"已完成：{idx+1}/{len(dataset)} 条")
    
    result = {
        "LLMs": MODEL_DISPLAY_NAME[model_name],
        "Method": "Self-StreamICL(ModelRoute+MemoryEvolve)",
        f"{dataset_name} Exact Match": calculate_exact_match(y_true, y_pred, task_type),
        f"{dataset_name} F1 Score": calculate_f1_score(y_true, y_pred, task_type),
        f"{dataset_name} MCC": calculate_mcc(y_true, y_pred, task_type)
    }
    
    print(f"评测：{result}")
    return result

def main():
    print("="*100)
    print("Self-StreamICL(ModelRoute+MemoryEvolve) 模型评测")
    print("="*100)
    
    all_results = []
    for model in TARGET_MODELS:
        model_result = {"LLMs": MODEL_DISPLAY_NAME[model], "Method": "Self-StreamICL(ModelRoute+MemoryEvolve)"}
        for dataset in TARGET_DATASETS:
            eval_result = evaluate_model(model, dataset)
            for key, value in eval_result.items():
                if key not in model_result:
                    model_result[key] = value
        all_results.append(model_result)
    
    final_df = pd.DataFrame(all_results)
    
    output_file = 'Self-StreamICL_评测结果.csv'
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*100)
    print("结果保存：", output_file)
    print("="*100)
    print("最终评测结果汇总：")
    print(final_df.to_string(index=False))
    return final_df

if __name__ == "__main__":
    final_result = main()

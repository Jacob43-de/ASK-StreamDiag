import os
import re
import json
import requests
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, matthews_corrcoef

API_KEY_QWEN = "sk-5a94d3c219a34ca3bccb3cd8ddeba635"
API_KEY_ZHIPU = "2059308a2b184f0ab74f251481a7eb76.aoUQ71oedYHTX4hj"
MAX_MEMORY = 16                # 记忆库最大容量
MEMORY_RETRIEVAL_TOPK = 3      # 每次检索最相似的记忆数量

DATA_CONFIG = {
    "DS-1000": {
        "path": r"C:\Users\liuxi\Desktop\streambench\data\ds_1000\test-00000-of-00001.parquet",
        "question_field": "question",
        "answer_field": "code",
        "eval_type": "code"
    },
    "DDXPlus": {
        "path": r"C:\Users\liuxi\Desktop\streambench\data\ddxplus\test-00000-of-00001 (1).parquet",
        "question_field": "symptoms",
        "answer_field": "diagnosis",
        "eval_type": "text"
    },
    "ToolBench": {
        "path": r"C:\Users\liuxi\Desktop\streambench\data\toolbench\test-00000-of-00001 (1).parquet",
        "question_field": "instruction",
        "answer_field": "tool_calls",
        "eval_type": "json"
    }
}

client_moonshot = OpenAI(
    api_key="sk-qYmUO15MCJ8hDB7LeAoIYxEKsrCCWzgjGnp4Dq3CEDoIh78a",
    base_url="https://api.moonshot.cn/v1",
)

def call_kimi(prompt: str) -> str:
    try:
        completion = client_moonshot.chat.completions.create(
            model="kimi-k2.6",
            messages=[{"role": "system", "content": "你是领域专家，只输出答案，不要解释。"},
                      {"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except:
        return ""

def call_qwen(prompt: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {API_KEY_QWEN}"}
        data = {
            "model": "qwen-turbo",
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": {"temperature": 0.0}
        }
        resp = requests.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                             headers=headers, json=data, timeout=30)
        return resp.json()["output"]["text"].strip()
    except:
        return ""

def call_zhipu(prompt: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {API_KEY_ZHIPU}"}
        data = {
            "model": "glm-4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        resp = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions",
                             headers=headers, json=data, timeout=30)
        return resp.json()["choices"][0]["message"]["content"].strip()
    except:
        return ""

def call_doubao(prompt: str) -> str:
    try:
        api_key_doubao = os.getenv('ark-c73d57b8-cbf1-4083-9b6e-05953b7b5574-18a7e')
        client = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=api_key_doubao)
        response = client.responses.create(
            model="doubao-seed-2-0-lite-260215",
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
        )
        return response.output.choices[0].message.content.strip()
    except:
        return ""

MODEL_FUNCS = {
    "Qwen": call_qwen,
    "Kimi": call_kimi,
    "Zhipu": call_zhipu,
    "DouBao": call_doubao
}

# ===================== 辅助函数 =====================
def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

def exact_match(pred: str, true: str) -> bool:
    return pred.strip().lower() == true.strip().lower()

def compute_f1(preds: List[str], trues: List[str], average='binary') -> float:
    all_chars = set(''.join(preds + trues))
    pred_vec = [1 if c in preds[0] else 0 for c in all_chars]
    true_vec = [1 if c in trues[0] else 0 for c in all_chars]
    return f1_score([true_vec], [pred_vec], average=average) if len(set(true_vec)) > 1 else 0.0

def compute_mcc(preds: List[str], trues: List[str]) -> float:
    all_chars = set(''.join(preds + trues))
    pred_vec = [1 if c in preds[0] else 0 for c in all_chars]
    true_vec = [1 if c in trues[0] else 0 for c in all_chars]
    return matthews_corrcoef(true_vec, pred_vec)

def evaluate_prediction(pred: str, true: str, eval_type: str) -> Dict[str, float]:
    if not pred or not true:
        return {"ACC": 0.0, "Exact_Match": 0.0, "F1": 0.0, "MCC": 0.0}
    
    # ACC
    acc = 1.0 if similarity(pred, true) >= 0.7 else 0.0
    
    # Exact Match
    em = 1.0 if exact_match(pred, true) else 0.0
    
    # F1 和 MCC
    f1 = compute_f1([pred], [true])
    mcc = compute_mcc([pred], [true])
    
    return {"ACC": acc, "Exact_Match": em, "F1": f1, "MCC": mcc}

def retrieve_memories(memory_list: List[Dict], query: str, k: int) -> List[Dict]:
    """检索与当前问题最相似的k个记忆"""
    similarities = [(mem, similarity(mem["question"], query)) for mem in memory_list]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [mem for mem, _ in similarities[:k]]

def evolve_memory(memory_list: List[Dict], new_item: Dict, max_size: int) -> List[Dict]:
    for i, mem in enumerate(memory_list):
        if similarity(mem["question"], new_item["question"]) > 0.9:
            memory_list[i] = new_item
            return memory_list
    memory_list.append(new_item)
    memory_list.sort(key=lambda x: similarity(x["question"], new_item["question"]), reverse=True)
    if len(memory_list) > max_size:
        memory_list.pop()
    return memory_list

# ===================== 主评测函数 =====================
def load_dataset(dataset_name: str):
    config = DATA_CONFIG[dataset_name]
    path = Path(config["path"])
    if not path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {path}")
    df = pd.read_parquet(path)
    data = []
    for _, row in df.iterrows():
        data.append({
            "question": str(row[config["question_field"]]),
            "answer": str(row[config["answer_field"]])
        })
    return data, config["eval_type"]

def run_evo_memory(model_name: str, dataset_name: str, data, eval_type: str) -> Dict:
    model_func = MODEL_FUNCS[model_name]
    memory = []          # 记忆库
    results = []
    
    for idx, item in enumerate(data):
        q = item["question"]
        true_a = item["answer"]
        
        # 1. 检索相关记忆
        retrieved = retrieve_memories(memory, q, MEMORY_RETRIEVAL_TOPK)
        memory_context = "\n".join([f"问题：{mem['question']}\n答案：{mem['answer']}" for mem in retrieved])
        
        # 2. 构造提示词（带记忆）
        prompt = f"""你是一个专业的AI助手。请根据以下参考示例，回答用户问题。
参考示例：
{memory_context if memory_context else "无"}
用户问题：{q}
请直接输出答案，不要解释。
"""
        pred_a = model_func(prompt)
        
        # 3. 评估
        scores = evaluate_prediction(pred_a, true_a, eval_type)
        
        # 4. 记忆进化：只存储预测正确的样本
        if scores["ACC"] == 1.0:
            evolve_memory(memory, {"question": q, "answer": pred_a}, MAX_MEMORY)
        
        results.append(scores)
        print(f"[{model_name}][{dataset_name}] 进度: {idx+1}/{len(data)} | 当前ACC: {scores['ACC']}")
    
    # 汇总指标
    avg_acc = sum(r["ACC"] for r in results) / len(results)
    avg_em = sum(r["Exact_Match"] for r in results) / len(results)
    avg_f1 = sum(r["F1"] for r in results) / len(results)
    avg_mcc = sum(r["MCC"] for r in results) / len(results)
    
    return {
        "ACC": avg_acc,
        "Exact_Match": avg_em,
        "F1": avg_f1,
        "MCC": avg_mcc
    }

def main():
    all_models = ["Qwen", "Kimi", "Zhipu", "DouBao"]
    datasets = ["DS-1000", "DDXPlus", "ToolBench"]
    
    final_results = []
    for model in all_models:
        for ds in datasets:
            print(f"\n========== 开始评测: 模型={model}, 数据集={ds} ==========")
            data, eval_type = load_dataset(ds)
            data = data[:1000]
            metrics = run_evo_memory(model, ds, data, eval_type)
            final_results.append({
                "Model": model,
                "Dataset": ds,
                "Method": "Evo-Memory",
                **metrics
            })
            print(f"结果: ACC={metrics['ACC']:.4f}, EM={metrics['Exact_Match']:.4f}, F1={metrics['F1']:.4f}, MCC={metrics['MCC']:.4f}")
    
    # 保存结果到CSV
    df = pd.DataFrame(final_results)
    df.to_csv("evo_memory_results.csv", index=False)
    print("\n所有结果已保存至 evo_memory_results.csv")
    print(df)

if __name__ == "__main__":
    main()
import os
import re
import json
import random
import requests
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, matthews_corrcoef

# ===================== 配置参数 =====================
API_KEY_QWEN = ""
API_KEY_ZHIPU = ""
NUM_SELF_CONSISTENCY = 5      # 自洽采样次数
GROUP_SIZE = 3                # 分组大小
TASK_SEQUENCE_LENGTH = 1000    # 每个数据集截取的任务数（终身学习序列长度）

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
    api_key="",
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
    acc = 1.0 if similarity(pred, true) >= 0.7 else 0.0
    em = 1.0 if exact_match(pred, true) else 0.0
    f1 = compute_f1([pred], [true])
    mcc = compute_mcc([pred], [true])
    return {"ACC": acc, "Exact_Match": em, "F1": f1, "MCC": mcc}

# ===================== 分组自洽机制 =====================
def group_self_consistency(model_func, question: str, num_samples: int, group_size: int) -> str:
    answers = []
    for _ in range(num_samples):
        # 加入温度扰动
        prompt = f"请回答以下问题，可以适当发挥创造力：\n{question}"
        ans = model_func(prompt)
        answers.append(ans)
    
    # 分组投票
    random.shuffle(answers)
    groups = [answers[i:i+group_size] for i in range(0, len(answers), group_size)]
    group_votes = []
    for group in groups:
        # 每组内用相似度聚合
        best_ans = max(set(group), key=lambda x: sum(similarity(x, other) for other in group))
        group_votes.append(best_ans)
    # 最终投票
    final = max(set(group_votes), key=lambda x: group_votes.count(x))
    return final

def run_lifelong_learning(model_name: str, dataset_name: str, data, eval_type: str) -> Dict:
    """
    模拟终身学习：任务以流式到达，模型需不断适应并保留知识。
    评测指标：所有任务的平均ACC/EM/F1/MCC。
    """
    model_func = MODEL_FUNCS[model_name]
    short_term_memory = []
    results = []
    
    for idx, item in enumerate(data[:TASK_SEQUENCE_LENGTH]):
        q = item["question"]
        true_a = item["answer"]
        pred_a = group_self_consistency(model_func, q, NUM_SELF_CONSISTENCY, GROUP_SIZE)
        
        scores = evaluate_prediction(pred_a, true_a, eval_type)
        results.append(scores)
        
        if scores["ACC"] == 1.0:
            short_term_memory.append({"question": q, "answer": pred_a})
            # 保留最近5个记忆
            if len(short_term_memory) > 5:
                short_term_memory.pop(0)
        
        print(f"[{model_name}][{dataset_name}] 任务 {idx+1}/{len(data[:TASK_SEQUENCE_LENGTH])} | ACC={scores['ACC']}")
    
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

def main():
    all_models = ["Qwen", "Kimi", "Zhipu", "DouBao"]
    datasets = ["DS-1000", "DDXPlus", "ToolBench"]
    
    final_results = []
    for model in all_models:
        for ds in datasets:
            print(f"\n========== 开始终身学习评测: 模型={model}, 数据集={ds} ==========")
            data, eval_type = load_dataset(ds)
            metrics = run_lifelong_learning(model, ds, data, eval_type)
            final_results.append({
                "Model": model,
                "Dataset": ds,
                "Method": "LifelongAgentBench",
                **metrics
            })
            print(f"结果: ACC={metrics['ACC']:.4f}, EM={metrics['Exact_Match']:.4f}, F1={metrics['F1']:.4f}, MCC={metrics['MCC']:.4f}")
    
    df = pd.DataFrame(final_results)
    df.to_csv("lifelong_agent_bench_results.csv", index=False)
    print("\n所有结果已保存至 lifelong_agent_bench_results.csv")
    print(df)

if __name__ == "__main__":
    main()

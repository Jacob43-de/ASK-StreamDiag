API_KEY_QWEN = ""
API_KEY_ZHIPU = ""
RUN_COUNT = 1000                # 运行前1000条数据
MAX_MEMORY_SIZE = 1000          # 记忆库最大容量（用于检索的样本数）
RETRIEVAL_K = 16                # 检索相似示例数量（论文ToolBench 用 16）
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 用于检索的嵌入模型

import requests
import os
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI
import re
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef
from sentence_transformers import SentenceTransformer
import faiss
import pickle

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

# 使用四个模型作为多个 agent
MODEL_LIST = [kimi, qwen, zhipu, doubao]
MODEL_NAMES = [m.__name__ for m in MODEL_LIST]

# ===================== 向量记忆库（共享记忆） =====================
class VectorMemory:
    def __init__(self, embedding_model_name=EMBEDDING_MODEL, max_size=MAX_MEMORY_SIZE):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.max_size = max_size
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.reset()
    def reset(self):
        """清空记忆"""
        self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度
        self.memory_items = []  # 存储 (question, sql) 元组
        self.question_embeddings = []
    def add(self, question, sql):
        """添加一个正确样本到记忆库"""
        if len(self.memory_items) >= self.max_size:
            # 移除最旧的
            self.memory_items.pop(0)
            self.question_embeddings.pop(0)
            # 重建索引
            self._rebuild_index()
        emb = self.embedder.encode(question, normalize_embeddings=True)
        self.memory_items.append((question, sql))
        self.question_embeddings.append(emb)
        self._add_to_index(emb)
    def _add_to_index(self, emb):
        """向 faiss 索引添加单个向量"""
        if self.index.ntotal == 0:
            self.index.add(emb.reshape(1, -1))
        else:
            self.index.add(emb.reshape(1, -1))
    def _rebuild_index(self):
        """重建索引（当删除旧项时）"""
        self.index = faiss.IndexFlatIP(self.dimension)
        if self.question_embeddings:
            all_embs = np.vstack(self.question_embeddings)
            self.index.add(all_embs)
    def retrieve(self, question, k=RETRIEVAL_K):
        """检索与当前问题最相似的 k 个示例"""
        if len(self.memory_items) == 0:
            return []
        emb = self.embedder.encode(question, normalize_embeddings=True).reshape(1, -1)
        distances, indices = self.index.search(emb, min(k, len(self.memory_items)))
        retrieved = []
        for idx in indices[0]:
            if idx != -1:
                retrieved.append(self.memory_items[idx])
        return retrieved
# 初始化共享记忆
shared_memory = VectorMemory()

def sql_similarity(pred_sql, true_sql):
    pred_clean = re.sub(r'\s+', ' ', pred_sql.strip().upper())
    true_clean = re.sub(r'\s+', ' ', true_sql.strip().upper())
    return SequenceMatcher(None, pred_clean, true_clean).ratio()
def exact_match(pred_sql, true_sql):
    pred_clean = re.sub(r'\s+', ' ', pred_sql.strip().upper())
    true_clean = re.sub(r'\s+', ' ', true_sql.strip().upper())
    return pred_clean == true_clean
def build_few_shot_prompt(question, examples):
    """构建带有检索示例的 prompt"""
    example_text = ""
    for ex_q, ex_sql in examples:
        example_text += f"用户问题：{ex_q}\nSQL语句：{ex_sql}\n\n"
    prompt = f"""以下是几个SQL生成示例：
{example_text}
请根据用户问题生成正确的SQL语句，只输出SQL，不要解释。
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

# ===================== MAM-StreamICL 评估 =====================
all_extracted_sqls = []
all_trues = []
all_model_names = []
all_valid = []

model_count = {name: 0 for name in MODEL_NAMES}
model_correct_sim = {name: 0 for name in MODEL_NAMES}
model_correct_exact = {name: 0 for name in MODEL_NAMES}
model_valid = {name: 0 for name in MODEL_NAMES}

for idx, item in enumerate(stream_data):
    # 轮换选择 agent
    agent_idx = idx % len(MODEL_LIST)
    model = MODEL_LIST[agent_idx]
    model_name = MODEL_NAMES[agent_idx]
    q = item["question"]
    true_sql = item["answer"]

    examples = shared_memory.retrieve(q, k=RETRIEVAL_K)
    prompt = build_few_shot_prompt(q, examples)
    
    pred_sql = model(prompt)
    # 判断有效输出
    if not pred_sql or len(pred_sql) < 3 or "无法回答" in pred_sql or "Unable" in pred_sql:
        valid = False
        pred_sql = ""
    else:
        valid = True
    # 计算相似度和精确匹配
    if valid:
        sim = sql_similarity(pred_sql, true_sql)
        is_correct_sim = sim > 0.7
        is_correct_exact = exact_match(pred_sql, true_sql)
    else:
        is_correct_sim = False
        is_correct_exact = False
    # 更新统计
    model_count[model_name] += 1
    if valid:
        model_valid[model_name] += 1
        if is_correct_sim:
            model_correct_sim[model_name] += 1
        if is_correct_exact:
            model_correct_exact[model_name] += 1
    
    # 记忆更新
    if valid and is_correct_sim:
        shared_memory.add(q, pred_sql)
    
    all_extracted_sqls.append(pred_sql)
    all_trues.append(true_sql)
    all_model_names.append(model_name)
    all_valid.append(valid)
    
    print(f"===== 第{idx+1}题 | 模型：{model_name} | 记忆库大小：{len(shared_memory.memory_items)} =====")
    print(f"问题：{q[:80]}...")
    print(f"模型回答：{pred_sql[:100] if pred_sql else '无效输出'}...")
    print(f"正确答案：{true_sql[:100]}...")
    print(f"有效输出：{valid} | 相似度正确：{is_correct_sim} | 精确匹配：{is_correct_exact}\n")

total_samples = len(stream_data)
valid_samples = sum(all_valid)

similarity_correct = [all_valid[i] and sql_similarity(all_extracted_sqls[i], all_trues[i]) > 0.7 
                      for i in range(total_samples)]
exact_match_correct = [all_valid[i] and exact_match(all_extracted_sqls[i], all_trues[i]) 
                       for i in range(total_samples)]

acc_similarity = np.mean(similarity_correct)
acc_exact = np.mean(exact_match_correct)

f1 = f1_score([1]*len(similarity_correct), similarity_correct, average='binary', zero_division=0)
mcc = matthews_corrcoef([1]*len(similarity_correct), similarity_correct)

# 各模型指标
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
print("MAM-StreamICL 评估结果（基于相似度阈值 0.7）")
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
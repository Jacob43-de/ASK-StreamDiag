# ===================== 配置参数 =====================
API_KEY_QWEN = ""
API_KEY_ZHIPU = ""
MAX_MEMORY = 16
MEMORY_EVOLVE = True          # 记忆进化开关
MODEL_EVALUATE = True         # 模型动态评分开关
ADAPTIVE_PROMPT = True        # 自适应提示词
RUN_COUNT = 1000               # 指定只运行前1000条数据

# 新增：细粒度诊断开关
FINE_GRAINED_DIAGNOSIS = True   # 是否启用子任务级诊断（Judge LLM）
MSU_PSK_ENABLED = True          # 是否启用MSU/PSK分类
P_NUM_VARIANTS = 2              # 提示扰动次数

import requests
import os
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI
import re
import copy
import random
from typing import List, Dict, Tuple

# ===================== 模型客户端初始化 =====================
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
model_score = {m.__name__: 1.0 for m in MODEL_LIST}
model_count = {m.__name__: 0 for m in MODEL_LIST}
model_correct = {m.__name__: 0 for m in MODEL_LIST}

# 新增：细粒度诊断统计
subtask_stats = {}  # {subtask_name: {"total":0, "correct":0, "msu":0, "psk":0, "reliable":0}}
msu_psk_stats = {"MSU": 0, "PSK": 0, "Reliable": 0}

total_count = 0
total_correct = 0
total_invalid = 0

# ===================== 辅助函数 =====================
def get_best_model():
    return max(MODEL_LIST, key=lambda m: model_score[m.__name__])

def update_model_score(model_name, is_correct):
    model_count[model_name] += 1
    if is_correct:
        model_correct[model_name] += 1
    acc = model_correct[model_name] / model_count[model_name]
    model_score[model_name] = acc

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

def sql_similarity(pred_sql, true_sql):
    pred_clean = re.sub(r'\s+', ' ', pred_sql.strip().upper())
    true_clean = re.sub(r'\s+', ' ', true_sql.strip().upper())
    return similarity(pred_clean, true_clean) > 0.7

# ===================== 子任务定义 =====================
SUBTASKS = [
    {"name": "table_identification", "description": "正确识别需要查询的表名，包括主表及关联表"},
    {"name": "filter_conditions", "description": "正确解析并写出 WHERE 子句中的过滤条件"},
    {"name": "field_selection", "description": "正确选择 SELECT 子句中需要的字段"},
    {"name": "join_logic", "description": "正确实现 JOIN 类型和连接条件（若需要多表关联）"}
]

# =====================Judge LLM (使用 Kimi 作为 Judge) =====================
def judge_subtask(question: str, generated_sql: str, subtask_desc: str) -> Tuple[bool, str]:
    """
    调用 Judge LLM 评估生成的 SQL 是否满足指定子任务的要求。
    返回 (是否成功, 原因)
    """
    judge_prompt = f"""你是一个严格的 SQL 评估专家。请判断以下生成的 SQL 是否满足指定的子任务要求。

用户问题：{question}
生成的 SQL：{generated_sql}
子任务描述：{subtask_desc}

请只输出 JSON 格式，不要有其他内容：
{{"success": true/false, "reason": "简短原因"}}
"""
    try:
        # 使用 Kimi 作为 Judge
        response = kimi(judge_prompt)
        # 解析 JSON
        import json
        # 尝试提取 JSON 部分
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("success", False), result.get("reason", "")
        else:
            return False, "Judge 输出格式错误"
    except Exception as e:
        return False, f"Judge 调用失败: {str(e)}"

def generate_prompt_variant(original_question: str, variant_type: str) -> str:
    """
    生成原问题的提示变体，用于 MSU/PSK 测试。
    variant_type: "rephrase" 或 "add_example"
    """
    if variant_type == "rephrase":
        prompt = f"""请将以下问题用不同的措辞重新表述一遍，保持语义完全相同，只输出改写后的问题：
原问题：{original_question}
改写后的问题："""
        new_q = qwen(prompt)  
        return new_q if new_q else original_question
    elif variant_type == "add_example":
        
        example = "例如查询用户表的所有字段：SELECT * FROM users;"
        return f"{original_question}\n{example}"
    else:
        return original_question

def classify_msu_psk(question: str, target_model, subtask_desc: str, original_sql: str) -> str:
    """
    对失败子任务进行 MSU/PSK 分类。
    通过生成 P 个提示变体，重新调用目标模型生成 SQL，再用 Judge 评估该子任务是否成功。
    计算成功率 r，根据阈值返回 'MSU', 'PSK', 'Reliable'
    """
    if not MSU_PSK_ENABLED:
        return "Unknown"
    
    success_count = 0
    total_tests = P_NUM_VARIANTS
    
    variant_types = ["rephrase", "add_example"]
    for i in range(total_tests):
        vtype = variant_types[i % len(variant_types)]
        variant_q = generate_prompt_variant(question, vtype)
        test_prompt = f"""你是SQL专家，根据用户问题生成正确的SQL语句。
用户问题：{variant_q}
输出要求：只输出SQL语句，不要解释。"""
        new_sql = target_model(test_prompt)
        if not new_sql:
            continue
        # 用 Judge 评估该子任务
        success, _ = judge_subtask(variant_q, new_sql, subtask_desc)
        if success:
            success_count += 1
    
    r = success_count / total_tests if total_tests > 0 else 0.0
    if r < 0.2:
        return "MSU"
    elif r < 0.8:
        return "PSK"
    else:
        return "Reliable"

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
    raise FileNotFoundError(f"未找到 {dataset_name}/{split}")

def load_parquet_data(dataset_name, split, field_map):
    file_path = get_parquet_path(dataset_name, split)
    df = pd.read_parquet(file_path)
    stream_data = []
    for _, row in df.iterrows():
        item = {
            "question": str(row[field_map["question"]]),
            "answer": str(row[field_map["answer"]])
        }
        stream_data.append(item)
    return stream_data[:RUN_COUNT]

stream_data = load_parquet_data(DATASET_NAME, SPLIT, FIELD_MAP)
shared_memory = []

# 初始化子任务统计
for sub in SUBTASKS:
    subtask_stats[sub["name"]] = {"total": 0, "correct": 0, "msu": 0, "psk": 0, "reliable": 0}

# ===================== 主循环 =====================
for idx, item in enumerate(stream_data):
    q = item["question"]
    true_a = item["answer"]
    total_count += 1
    
    # 1. 原子子任务分解
    step1_prompt = f"""
    任务：将用户问题拆解为SQL生成所需的子步骤：
    用户问题：{q}
    请输出：
    1. 需要查询的表名
    2. 过滤条件
    3. 需要的字段
    4. 是否需要JOIN
    """
    decomposition = qwen(step1_prompt)
    
    # 2. 自适应模型路由
    if "JOIN" in decomposition.upper():
        model = kimi if MODEL_EVALUATE else MODEL_LIST[idx % len(MODEL_LIST)]
    else:
        model = qwen if MODEL_EVALUATE else MODEL_LIST[idx % len(MODEL_LIST)]
    
    # 3. 构造带记忆的SQL生成提示
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
    pred_a = model(prompt)
    
    # 校验是否生成了有效的SQL
    if not pred_a or "无法回答" in pred_a or "Unable" in pred_a:
        total_invalid += 1
        print(f"===== 第{idx+1}题 | 模型：{model.__name__} =====")
        print(f"问题：{q[:60]}...")
        print(f"模型回答：{pred_a}")
        print(f"正确答案：{true_a}")
        print(f"是否正确：False（模型未输出有效SQL）")
        print(f"记忆库：{len(shared_memory)} | 模型得分：{model_score[model.__name__]:.2f}\n")
        continue
    
    # 4. 整体正确性判断（用于记忆进化和模型评分）
    is_correct = sql_similarity(pred_a, true_a)
    if is_correct:
        total_correct += 1
    
    # 5. 细粒度子任务诊断（Judge LLM）
    subtask_results = []
    if FINE_GRAINED_DIAGNOSIS:
        for sub in SUBTASKS:
            subtask_stats[sub["name"]]["total"] += 1
            success, reason = judge_subtask(q, pred_a, sub["description"])
            subtask_results.append((sub["name"], success, reason))
            if success:
                subtask_stats[sub["name"]]["correct"] += 1
            else:
                # 对失败子任务进行 MSU/PSK 分类
                msu_psk_type = classify_msu_psk(q, model, sub["description"], pred_a)
                if msu_psk_type == "MSU":
                    subtask_stats[sub["name"]]["msu"] += 1
                    msu_psk_stats["MSU"] += 1
                elif msu_psk_type == "PSK":
                    subtask_stats[sub["name"]]["psk"] += 1
                    msu_psk_stats["PSK"] += 1
                elif msu_psk_type == "Reliable":
                    subtask_stats[sub["name"]]["reliable"] += 1
                    msu_psk_stats["Reliable"] += 1
    
    # 更新模型评分
    if MODEL_EVALUATE:
        update_model_score(model.__name__, is_correct)
    
    # 记忆进化：只存整体正确的样本
    if MEMORY_EVOLVE:
        if is_correct:
            memory_evolve(shared_memory, {"q": q, "a": pred_a})
    else:
        if is_correct:
            shared_memory.append({"q": q, "a": pred_a})
            if len(shared_memory) > MAX_MEMORY:
                shared_memory.pop(0)
    
    # 输出信息
    print(f"===== 第{idx+1}题 | 模型：{model.__name__} =====")
    print(f"问题：{q[:60]}...")
    print(f"模型回答：{pred_a}")
    print(f"正确答案：{true_a}")
    print(f"整体是否正确：{is_correct}")
    if FINE_GRAINED_DIAGNOSIS:
        print("细粒度子任务诊断结果：")
        for name, succ, reason in subtask_results:
            status = "✓ 成功" if succ else "✗ 失败"
            print(f"  - {name}: {status} ({reason})")
    print(f"记忆库大小：{len(shared_memory)} | 模型得分：{model_score[model.__name__]:.2f}\n")

print("="*80)
print("最终准确率（ACC）及细粒度诊断统计")
print("="*80)
valid_count = total_count - total_invalid
overall_acc = total_correct / valid_count if valid_count > 0 else 0.0
print(f"\n整体统计：")
print(f"   总运行题目数：{total_count}")
print(f"   有效回答数：{valid_count}")
print(f"   无效回答数：{total_invalid}")
print(f"   正确回答数：{total_correct}")
print(f"   整体准确率（ACC）：{overall_acc:.4f} ({total_correct}/{valid_count})")

print(f"\n各模型详细得分：")
for model_name in model_score:
    if model_count[model_name] > 0:
        acc = model_score[model_name]
        print(f"   {model_name:8s} | 答题数：{model_count[model_name]:3d} | 正确数：{model_correct[model_name]:3d} | 准确率：{acc:.4f}")
    else:
        print(f"   {model_name:8s} | 答题数：{model_count[model_name]:3d} | 正确数：{model_correct[model_name]:3d} | 准确率：N/A")

if FINE_GRAINED_DIAGNOSIS:
    print("\n原子子任务级诊断统计：")
    for sub in SUBTASKS:
        stats = subtask_stats[sub["name"]]
        total_sub = stats["total"]
        if total_sub > 0:
            acc_sub = stats["correct"] / total_sub
            print(f"   {sub['name']:20s} | 总数：{total_sub:4d} | 正确数：{stats['correct']:4d} | 准确率：{acc_sub:.4f}")
            print(f"        MSU: {stats['msu']:3d} | PSK: {stats['psk']:3d} | Reliable: {stats['reliable']:3d}")
    
    print("\nMSU/PSK 总体分布（所有失败子任务）:")
    total_fail_sub = sum(msu_psk_stats.values())
    print(f"   MSU: {msu_psk_stats['MSU']} ({msu_psk_stats['MSU']/total_fail_sub*100:.1f}%)")
    print(f"   PSK: {msu_psk_stats['PSK']} ({msu_psk_stats['PSK']/total_fail_sub*100:.1f}%)")
    print(f"   Reliable: {msu_psk_stats['Reliable']} ({msu_psk_stats['Reliable']/total_fail_sub*100:.1f}%)")

print("\n" + "="*80)
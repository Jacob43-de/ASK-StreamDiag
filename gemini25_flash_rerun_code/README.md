# Gemini 2.5 Flash 重跑实验代码说明

本目录包含将 ASK-StreamDiag 论文实验的目标模型替换为 **Gemini 2.5 Flash** 后的全量重跑代码。

> **红线声明**：本目录所有代码均不含任何硬编码 API key；未配置环境变量时不会发起真实网络请求。dry-run 产物中的数值全部来自 mock 标注，**不得用于论文**。

---

## 目录结构

```
gemini25_flash_rerun/
├── common/                          # 公共模块
│   ├── __init__.py
│   ├── gemini_client.py             # Gemini 2.5 Flash HTTP 客户端（重试/超时/代理）
│   ├── openai_clients.py            # GPT-4 judge + GPT-3.5-turbo dispatcher
│   ├── data_loader.py               # 统一数据加载（自动探测 test parquet）
│   ├── scorer.py                     # 答案抽取 + 判分 + 指标计算
│   └── prompts.py                   # 三个数据集领域 prompt + 子任务模板
├── run_zero_shot.py                 # Zero-Shot
├── run_few_shot.py                  # Few-Shot (k=3)
├── run_cot.py                       # Chain-of-Thought
├── run_growprompt.py                # GrowPrompt (window=16)
├── run_self_streamicl.py            # Self-StreamICL (M=16, memory evolve)
├── run_mam_streamicl.py             # MAM-StreamICL (top-k=16 retrieval)
├── run_evo_memory.py                # Evo Memory (M=16, top-k=3)
├── run_lifelong_agent_bench.py      # LifelongAgentBench (self-consistency=5, group=3)
├── ask_streamdiag_gemini.py         # ASK-StreamDiag 主管线
├── run_all_gemini25.py              # 一键跑全部 8 方法 × 3 数据集
├── test_dry_run.py                  # 离线自测脚本（mock，不调真实 API）
├── requirements.txt
└── README.md                        # 本文件
```

---

## 方法与原脚本对应关系

| 新脚本 | 原脚本 | 说明 |
|--------|--------|------|
| `run_zero_shot.py` | `01 zero-shot.py` | prompt 结构一致，目标模型换 Gemini 2.5 Flash |
| `run_few_shot.py` | `02 few-shot.py` / `few-shot(Gemini).py` | 3-shot，示例按数据集领域适配 |
| `run_cot.py` | `03 CoT.py` / `COT(Gemini).py` | CoT 推理 prompt，代码块抽取 |
| `run_growprompt.py` | `Growprompt.py` / `Growprompt(Gemini).py` | 滑动窗口 16，带正确性反馈 |
| `run_self_streamicl.py` | `04 Self-StreamICL.py` / `Self-StreamICL（Gemini）.py` | M=16，memory evolve，相似度阈值 0.9 |
| `run_mam_streamicl.py` | `05 MAM-StreamICL.py` / `MAM-StreamICL(Gemini).py` | top-k=16 检索，**检索方式改为 SequenceMatcher**（原用 FAISS+BGE，环境未装） |
| `run_evo_memory.py` | `evo_memory_eval.py` | M=16, top-k=3 检索，相似度进化 |
| `run_lifelong_agent_bench.py` | `lifelong_agent_bench_eval.py` | 自洽采样 5 次，分组大小 3，短期记忆 5 |
| `ask_streamdiag_gemini.py` | `Ablation and Analysis experiment.py` + `Self-StreamICL（Gemini）.py` | 完整 ASK 管线，见下方详述 |

---

## 数据集字段映射决策

基于实测 parquet schema（2026-10-17 验证）：

| 数据集 | 行数 | 问题字段 | 金标答案字段 | eval_type | 备注 |
|--------|------|----------|-------------|-----------|------|
| ds_1000 | 1000 | `prompt` | `reference_code` | code | 还有 `metadata`、`code_context` 列 |
| ddxplus | 1764（取前1000） | `PATIENT_PROFILE` | `PATHOLOGY` | text | 还有 `EVIDENCES_ENG`、`DIFFERENTIAL_DIAGNOSIS` 等 |
| toolbench | 750（全部使用） | `query` | `api_name` | json | 辅助金标：`api_input`（JSON 字符串，完整入参） |

**映射依据**：
- 旧脚本中 `FIELD_MAP`（question/SQL、symptoms/diagnosis、instruction/tool_calls）与真实 parquet 不符，全部废弃。
- toolbench 只有 750 条，全部使用（limit=1000 自动截断到 750）。
- ddxplus 有 1764 条，取前 1000 条（RUN_COUNT=1000）。

---

## 指标口径（严格复刻原脚本）

- **ACC（Post-repair Accuracy）**：
  - code 类：`SequenceMatcher(normalize_code(pred), normalize_code(gold)).ratio() > 0.7`
    - normalize_code = upper() + 空白折叠
  - text 类：`SequenceMatcher(pred.strip().lower(), gold.strip().lower()).ratio() > 0.7`
  - json 类：API 名精确匹配（lower 后比较）
- **EM（Exact Match）**：归一化后字符串完全相等
- **F1**：`f1_score([1]*n, correctness_bools, average='binary', zero_division=0)`
  - **注意**：这是原脚本的特殊写法——y_true 全为 1（因为金标是"全对"），y_pred 是每条样本的对错布尔值
- **MCC**：`matthews_corrcoef([1]*n, correctness_bools)`

---

## ASK-StreamDiag 管线详情

### 角色配置（论文 §5.4）
- **目标模型**：Gemini 2.5 Flash，temperature=0.0
- **Dispacher（子任务分解）**：GPT-3.5-turbo，temperature=0.0
- **Judge（逐子任务评判）**：GPT-4，temperature=0.2，系统提示"你是一个严格的评估专家。请只输出 JSON 格式。"

### 超参数
- M（记忆大小）= 16
- P（提示扰动次数）= 4（论文 3-4，消融脚本用 5，做成 `--p-variants` 参数可调）
- τ_low = 0.2，τ_high = 0.8

### 真实路由修复（替换原脚本的模拟逻辑）
原 `Ablation and Analysis experiment.py` 中路由修复用 `random.random() < 0.9` / `< 0.7` 模拟，本代码全部替换为真实 API 调用：
- **MSU**：交给 GPT-4 直接回答该子任务（真实调用 `gpt-4`）
- **PSK**：用 GPT-3.5-turbo 改写问题后，让 Gemini 2.5 Flash 重试（真实调用）
- **Reliable**：保持原答案不修复

### 三数据集子任务模板
- **DS-1000（代码）**：problem_understanding / dataframe_structure / pandas_operations / code_correctness
- **DDXPlus（诊断）**：symptom_identification / differential_diagnosis / diagnostic_reasoning / final_diagnosis
- **ToolBench（工具）**：intent_recognition / api_selection / parameter_extraction / argument_format

---

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `GEMINI_API_KEY` | 是 | Google Gemini API 密钥 |
| `OPENAI_API_KEY` | 是（ASK 管线需要） | OpenAI API 密钥（GPT-4 judge + GPT-3.5 dispatcher） |
| `OPENAI_BASE_URL` | 否 | 自定义 OpenAI 兼容端点（如代理/中转） |
| `HTTPS_PROXY` | 否 | 标准环境变量代理（requests 默认读取，请勿硬编码） |
| `HTTP_PROXY` | 否 | 同上 |

---

## 运行命令

### 单个方法单个数据集
```powershell
$py = "C:\Users\liuxi\AppData\Local\Programs\Python\Python314\python.exe"
cd C:\Users\liuxi\Desktop\streambench\gemini25_flash_rerun

# Zero-Shot on ds_1000
& $py run_zero_shot.py --dataset ds_1000 --limit 1000 --output-dir .

# ASK-StreamDiag on toolbench
& $py ask_streamdiag_gemini.py --dataset toolbench --limit 750 --output-dir .
```

### 一键跑全部
```powershell
& $py run_all_gemini25.py --output-dir . --limit 1000
```

### 离线自测（不调真实 API）
```powershell
& $py test_dry_run.py
```

---

## 输出文件

```
raw_outputs/
├── zero_shot_ds_1000.jsonl
├── zero_shot_ddxplus.jsonl
├── zero_shot_toolbench.jsonl
├── few_shot_ds_1000.jsonl
├── ...（每个方法 × 每个数据集一个 jsonl）
└── ask_streamdiag_toolbench.jsonl

table6_gemini25_flash.csv    # 最终汇总表，列结构同论文 Table 6
```

每条 jsonl 记录包含：`index`、`method`、`dataset`、`question`、`gold`、`raw_response`、`extracted_answer`、`valid`、`acc_correct`、`em_correct`、时间戳、耗时等。

**断点续跑**：已完成的样本 index 自动跳过，每条写盘后 flush。

---

## 预计调用量估算

按每样本调用次数：

| 方法 | 每样本 Gemini 调用 | 每样本 GPT-4 调用 | 每样本 GPT-3.5 调用 | 备注 |
|------|-------------------|-------------------|---------------------|------|
| Zero-Shot | 1 | 0 | 0 | |
| Few-Shot | 1 | 0 | 0 | |
| CoT | 1 | 0 | 0 | |
| GrowPrompt | 1 | 0 | 0 | |
| Self-StreamICL | 1 | 0 | 0 | |
| MAM-StreamICL | 1 | 0 | 0 | |
| Evo Memory | 1 | 0 | 0 | |
| LifelongAgentBench | 5 | 0 | 0 | 自洽采样 5 次 |
| ASK-StreamDiag | 1 + P×(1+1)×失败子任务数 | 1×子任务数 + P×失败子任务数 | 1 + P×失败子任务数 | 最坏情况约 1+4×4=17 次 Gemini |

**总量估算**（8 方法 × 3 数据集 × ~917 样本均值）：
- 简单方法（7 个）：7 × 3 × 1000 ≈ 21,000 次 Gemini 调用
- LifelongAgentBench：3 × 1000 × 5 = 15,000 次 Gemini 调用
- ASK-StreamDiag：3 × 1000 × ~10 次 ≈ 30,000 次 Gemini + 12,000 次 GPT-4 + 12,000 次 GPT-3.5
- **总计：约 66,000 次 Gemini + 12,000 次 GPT-4 + 12,000 次 GPT-3.5**

### Table 7 预算对照额外调用量
- Binary+GPT4-Judge 真实对照：约 3 × 1000 × ~30% 失败率 ≈ **900 次 GPT-4 + 900 次 Gemini 重试**

---

## Table 7 预算对照（run_budget_controls.py）

论文 Table 7（tab:budget_matched）每个目标模型 4 行：

| 行 | 类型 | 说明 |
|----|------|------|
| Binary+GPT4-Routing | **公式派生** | `routing = zs + (1 - zs) × 0.734`，其中 zs = Table 6 Zero-Shot 行 ACC |
| Binary+GPT4-Judge | **真实对照运行** | MAM 记忆基线 + GPT-4 二元评判（无子任务分解），fail 时重试 1 次 |
| Oracle-Routing | **公式派生** | `oracle = zs + (1 - zs) × 0.815` |
| ASK-StreamDiag | 直接取 Table 6 ASK 行 ACC | |

**公式已验证**：对 Kimi/GPT-3.5/Gemini 旧 Table 7 共 9 个单元格逐格验算，精确到小数点后 3 位，与论文数值完全一致。

**Binary+GPT4-Judge 实现口径**：
- 底座 = MAM-StreamICL 记忆机制（top-k=16 检索，只存正确样本）
- GPT-4 只输出任务级二元 `{"pass": true/false}`，不做子任务分解
- 判 fail 时，追加二元反馈让 Gemini 2.5 Flash 重试 1 次，取重试结果
- 不做 MSU/PSK 分类、不路由到 GPT-4 作答

> **注意**：Binary+GPT4-Judge 是按论文 §5.10.7(1) 文字重建的控制实验（原工程未找到对应脚本）。**数值使用前须与导师确认控制实现是否一致。** 原始响应落盘在 `raw_outputs/budget_binary_gpt4_judge_*.jsonl` 可复查。

运行方式：
```powershell
# 只算公式行（不调 API）
& $py run_budget_controls.py --output-dir . --formula-only

# 完整跑（含 Binary+GPT4-Judge 真实对照）
& $py run_budget_controls.py --output-dir . --limit 1000
```

---

## 与原脚本的差异说明（非静默优化）

1. **MAM-StreamICL 检索方式**：原脚本用 FAISS + BAAI/bge-base-en-v1.5 向量检索，本代码改用 SequenceMatcher 字符串相似度检索（环境未装 sentence-transformers/faiss）。核心逻辑（top-k 检索 + few-shot prompt + 只存正确样本）保持一致。
2. **LifelongAgentBench 温度**：原脚本在 prompt 中加"可以适当发挥创造力"做温度扰动；本代码直接用 temperature=0.3 采样 5 次。
3. **字段映射**：全部基于实测 parquet schema 重建，废弃旧 FIELD_MAP。
4. **路由修复（MSU 真实 API 化）**：原 Ablation 脚本用 `random.random() < 0.9` / `< 0.7` 模拟路由成功率。本代码将 MSU 修复重构为真实 API 调用：GPT-4 针对失败子任务直接生成修复答案，替换最终输出后重新整体判分。若真实重跑后 Table 6 口径与导师预期不一致，可据 raw_outputs 中 `repair_applied` 与 `msu_psk_log` 字段调整实现。
5. **P_NUM_VARIANTS**：消融脚本用 5，论文写 3-4，本代码默认 4，可通过 `--p-variants` 调整。
6. **ASK 记忆滑动窗口**：正确样例写入后始终保留最近 M=16 条（`memory[-16:]`），与其他流式方法口径一致。

---

## 待真实运行时确认的假设

1. Gemini 2.5 Flash 默认开启 thinking，响应 `candidates[0].content.parts` 中会先出现带 `thought: true` 标记的推理块。客户端已自动剥离思考块、只拼接正式答案 part，不计入判分。如未来官方响应字段名有变动，以实际 raw json 为准（原始响应已落盘到 raw_outputs 可复查）。
2. GPT-4 / GPT-3.5-turbo 在 `OPENAI_BASE_URL` 自定义端点下的模型名是否需要调整（如中转平台的模型名映射）。
3. toolbench 的 `api_input`（JSON 字符串）是否需要纳入判分（当前只判 `api_name` 匹配，`api_input` 作为辅助金标存储但未参与判分——与原脚本口径一致）。
4. ddxplus 的诊断文本是否需要做同义词归一化（当前用 strip().lower() + 0.7 相似度阈值）。

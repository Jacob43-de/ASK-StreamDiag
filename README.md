# ASK-StreamDiag

**Fine-Grained Streaming Failure Diagnosis for LLMs via Atomic Subtasks and Prompt-Perturbation Knowledge Boundaries**

Companion code for the manuscript *"ASK-StreamDiag: Fine-Grained Streaming Failure Diagnosis for LLMs via Atomic Subtasks and Prompt-Perturbation Knowledge Boundaries"* (submitted to **Mathematics**, MDPI, 2026).

**ASK-StreamDiag** replaces coarse-grained binary pass/fail evaluation of streaming LLM agents with a fine-grained diagnostic pipeline. Each task is decomposed into domain-specific **atomic subtasks**, a judge LLM audits every subtask independently, and each failed subtask is classified as either **MSU** (model-specific unknown — the failure persists across tested prompt variants) or **PSK** (prompt-sensitive known — the failure varies across variants), enabling targeted adaptive routing and repair instead of blindly upgrading every failure to a stronger model.

---

## Table of Contents

- [Highlights](#highlights)
- [Pipeline overview](#pipeline-overview)
- [Repository structure](#repository-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [Scoring metrics](#scoring-metrics)
- [Reproducing the tables and figures](#reproducing-the-tables-and-figures)
- [Main results](#main-results)
- [Hyperparameters](#hyperparameters)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)

---

## Highlights

- **81.25% per-subtask diagnostic accuracy** against human-annotated gold labels (300 labelled failed instances, Cohen's $\kappa$ and Krippendorff's $\alpha$ reported in the paper).
- **Task completion rate raised from 70.0% (binary feedback) to 88.5%**, and an **81.5% post-failure repair success rate at only 1.6x the baseline inference cost** (vs. 73.4% at 3.2x for routing every failure to a stronger model).
- Evaluated on **three public benchmarks** (DS-1000, DDXPlus, ToolBench) and **six target LLMs** (Qwen-turbo, Kimi-k2.6, Zhipu GLM-4, DouBao-seed-2.0-lite, GPT-3.5-turbo, Gemini-2.5-flash), against **seven baseline methods** (Zero Shot, Few Shot, CoT, GrowPrompt, MAM-StreamICL, Evo-Memory, LifelongAgentBench).
- Complete budget-matched controls (Table 7), MSU/PSK repair analysis (Section 5.10), threshold/granularity/judge-round ablations, and cross-domain generalization experiments.

---

## Pipeline overview

The framework (paper Sections 3–4) runs as four sequential stages:

1. **Atomic Subtask Decomposition** — a dispatcher (GPT-3.5-turbo) selects and instantiates the relevant subtasks from a fixed, pre-defined domain taxonomy. The taxonomy is fixed before model inference and is not learned from the models being evaluated.
2. **Subtask Judge Evaluation** — a single judge LLM (GPT-4, temperature 0.2) independently scores each subtask and emits structured JSON diagnostic feedback (correctness + error attribution).
3. **Knowledge Boundary Classification** — failed subtasks are re-tested under `P` prompt perturbations (instruction paraphrases, added few-shot exemplars, format constraints). A success rate at or below `tau_low = 0.2` is labelled **MSU**; a rate at or above `tau_high = 0.8` is labelled **PSK**; intermediate cases are labelled reliable/uncertain.
4. **Adaptive Routing / Repair** — MSU failures are routed to a stronger model; PSK failures are repaired by prompt rewriting. Verified solved exemplars enter a streaming memory buffer of size `M = 16`.

> **Scope note (paper Sections 4.2 and 5.8).** MSU/PSK are operational labels: they mean "persistent under the tested perturbation budget" versus "prompt-sensitive within that budget", not a claim that the model's parameters contain no knowledge whatsoever. The implementation uses a **single** judge; the Dawid–Skene multi-judge equation in Section 4 is included only as a theoretical reference.

---

## Repository structure

```
ASK-StreamDiag/
├── data/                                  # Benchmark datasets (parquet, not committed)
│   ├── ds_1000/test-*.parquet
│   ├── ddxplus/test-*.parquet
│   └── toolbench/test-*.parquet
│
├── 01 zero-shot.py                        # Zero Shot baseline
├── 02 few-shot.py                         # Few Shot baseline (k=3)
├── 03 CoT.py                              # Chain-of-Thought baseline
├── 04 Self-StreamICL.py                   # Self-StreamICL (M=16, memory evolution)
├── 05 MAM-StreamICL.py                    # MAM-StreamICL (top-k=16 retrieval, FAISS+BGE)
├── Growprompt.py                          # GrowPrompt streaming baseline
├── evo_memory_eval.py                     # Evo-Memory baseline
├── lifelong_agent_bench_eval.py           # LifelongAgentBench baseline
├── Ablation and Analysis experiment.py    # Full ASK-StreamDiag pipeline + ablations
│
├── * (GPT).py / * (Gemini).py             # Per-target-model variants of the baselines
│
├── 新增分析实验1.py                        # Error-distribution analysis figure
├── 新增分析实验2.py                        # MSU/PSK repair-effectiveness figure
├── 新增分析实验3.py                        # Cost–diagnostic-depth trade-off figure
├── 画图代码1.py / 画图代码3.py / 画图4.py / 画图5.py   # Paper Figures 3/4/6/7
│
├── generate_results.py                    # Aggregates recorded runs into result CSVs/TXTs
│
├── gemini25_flash_rerun/                  # Clean re-implementation (Gemini-2.5-flash target)
│   ├── common/
│   │   ├── gemini_client.py               # Gemini HTTP client (retry / timeout / proxy)
│   │   ├── openai_clients.py              # GPT-4 judge + GPT-3.5 dispatcher clients
│   │   ├── data_loader.py                 # Unified parquet loader
│   │   ├── scorer.py                      # Answer extraction + ACC/EM/F1/MCC scoring
│   │   └── prompts.py                     # Domain prompt and subtask templates
│   ├── run_zero_shot.py / run_few_shot.py / run_cot.py
│   ├── run_growprompt.py / run_self_streamicl.py / run_mam_streamicl.py
│   ├── run_evo_memory.py / run_lifelong_agent_bench.py
│   ├── ask_streamdiag_gemini.py           # Full ASK-StreamDiag pipeline
│   ├── run_budget_controls.py             # Table 7 budget-matched controls
│   ├── run_all_gemini25.py                # One-click 8 methods x 3 datasets
│   └── test_dry_run.py                    # Offline self-test (mock, no API calls)
│
├── results/                               # Result tables (Table 6 / Table 7, ...)
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Domain | Paper samples | Atomic subtasks | Evaluated by |
|---|---|---|---|---|
| **DS-1000** | Python data-science code generation | 1,000 | 4 | ACC / EM / F1 / MCC |
| **DDXPlus** | Clinical differential diagnosis | 1,200 | 5 | ACC / EM / F1 / MCC |
| **ToolBench** | Tool / API invocation | 800 | 4 | ACC / EM / F1 / MCC |

**Atomic subtask taxonomy (paper Section 5.1):**

- **DS-1000 (4):** algorithmic comprehension, control-flow rationality, variable/data-structure specification, grammatical syntax / output-format compliance.
- **DDXPlus (5):** symptom comprehension, differential disease enumeration, risk-factor weighting, definitive diagnosis determination, diagnostic justification validity.
- **ToolBench (4):** user-intent identification, parameter-value extraction, standardized calling syntax, returned-content semantic parsing.

**Parquet schema used by the loader (`common/data_loader.py`):**

| Dataset | Question column | Gold column | Eval type | Notes |
|---|---|---|---|---|
| `ds_1000` | `prompt` | `reference_code` | `code` | also `metadata`, `code_context` |
| `ddxplus` | `PATIENT_PROFILE` | `PATHOLOGY` | `text` | also `EVIDENCES_ENG`, `DIFFERENTIAL_DIAGNOSIS` |
| `toolbench` | `query` | `api_name` | `json` | `api_input` stored as auxiliary gold |

Download the three public benchmarks from their official releases (DS-1000: Lai et al. 2022; DDXPlus: Van Ness et al., NeurIPS 2022 Datasets & Benchmarks; ToolBench: Qin/Tang et al. 2023), convert/keep them in the test-parquet layout above, and place them under `data/`.

---

## Installation

```bash
git clone https://github.com/Jacob43-de/ASK-StreamDiag.git
cd ASK-StreamDiag
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate
python -m pip install -r requirements.txt
```

Tested with Python 3.10–3.14. The core requirements are `requests`, `openai`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `seaborn`. The original `05 MAM-StreamICL.py` additionally needs `faiss-cpu` and `sentence-transformers` (listed, commented out, in `requirements.txt`; the `gemini25_flash_rerun/` re-implementation uses SequenceMatcher retrieval and does not need them).

---

## Configuration

No API key is hard-coded anywhere. Keys are read from environment variables.

| Variable | Required for | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | GPT-4 judge, GPT-3.5 dispatcher, GPT-3.5 target | OpenAI-compatible API key |
| `OPENAI_BASE_URL` | optional | Custom OpenAI-compatible endpoint (proxy / relay) |
| `GEMINI_API_KEY` | Gemini-2.5-flash target | Google Gemini API key |
| `QWEN_API_KEY` | Qwen-turbo target | DashScope API key |
| `MOONSHOT_API_KEY` | Kimi-k2.6 target | Moonshot API key |
| `ZHIPU_API_KEY` | GLM-4 target | Zhipu API key |
| `ARK_API_KEY` | DouBao-seed-2.0-lite target | Volcano Engine ARK key |
| `HTTP_PROXY` / `HTTPS_PROXY` | optional | Standard proxy variables, read by `requests` |

PowerShell example:

```powershell
$env:OPENAI_API_KEY = ""
$env:GEMINI_API_KEY = ""
```

---

## Quick start

### Offline self-test (no API calls, no cost)

```powershell
cd gemini25_flash_rerun
python test_dry_run.py
```

### Single method on a single dataset

```powershell
cd gemini25_flash_rerun
python run_zero_shot.py --dataset ds_1000 --limit 1000 --output-dir .
python ask_streamdiag_gemini.py --dataset toolbench --limit 750 --output-dir .
```

### All eight methods on all three datasets

```powershell
python run_all_gemini25.py --output-dir . --limit 1000
```

### Budget-matched controls (paper Table 7)

```powershell
# Formula-derived rows only, no API calls:
python run_budget_controls.py --output-dir . --formula-only
# Full run including the Binary+GPT4-Judge control:
python run_budget_controls.py --output-dir . --limit 1000
```

### Original scripts (other target LLMs)

The original scripts at the repository root each target one model family (Qwen/Kimi/Zhipu/DouBao by default, or the `(GPT)` / `(Gemini)` variants):

```bash
python "01 zero-shot.py"
python "02 few-shot.py"
python "03 CoT.py"
python "04 Self-StreamICL.py"
python "05 MAM-StreamICL.py"
python "Ablation and Analysis experiment.py"
```

**Outputs:** per-task raw predictions are written to `raw_outputs/<method>_<dataset>.jsonl` with fields `index`, `method`, `dataset`, `question`, `gold`, `raw_response`, `extracted_answer`, `valid`, `acc_correct`, `em_correct`, and timing. Every record is flushed immediately, so completed indices are skipped automatically when a run is resumed. Aggregated tables are written to `table6_gemini25_flash.csv` / `table7_gemini25_flash.csv`.

---

## Scoring metrics

The metrics follow paper Section 5.6 and are implemented in `gemini25_flash_rerun/common/scorer.py`.

- **ACC (accuracy).** Answers are normalized before comparison:
  - *code* (`DS-1000`): fenced code block extraction (falling back to `SELECT`/`def` heuristics), uppercasing, whitespace collapsing; correct when `SequenceMatcher(pred, gold).ratio() > 0.7`.
  - *text* (`DDXPlus`): `strip().lower()` then the same `SequenceMatcher` similarity threshold of 0.7.
  - *json* (`ToolBench`): the predicted `api_name` is extracted and compared with the gold API name after lower-casing — exact API-name match.
- **EM (exact match).** The normalized prediction is byte-identical to the normalized reference (`DS-1000`, `DDXPlus`), or the predicted API name equals the gold API name (`ToolBench`). EM is the stricter metric: it is always **bounded above by ACC** on the same outputs.
- **F1 / MCC.** Macro F1 and the multi-class Matthews correlation coefficient over answer classes, as reported in the paper tables.

> **Scoring alignment note.** The archived `common/scorer.py` computes F1/MCC with the all-positive-reference convention (`f1_score([1]*n, correctness)` / `matthews_corrcoef([1]*n, correctness)`), which matches the original run logs' binary-correctness computation but does not emit the multi-class MCC of Section 5.6. Align this function to the multi-class answer-label implementation before relying on the code for the F1/MCC columns; ACC and EM are unaffected.

---

## Reproducing the tables and figures

| Paper element | Source script / file |
|---|---|
| Table 6 (main results, 8 methods x 6 models x 3 datasets) | `run_all_gemini25.py` (Gemini target) / root scripts (other models); `results/main_experiment_results.csv`; `generate_results.py` aggregates recorded runs |
| Table 7 (budget-matched controls) | `gemini25_flash_rerun/run_budget_controls.py`; `results/table7_gemini25_flash.csv` |
| Tables 8–9 (raw pre-repair ACC / EM) | Zero-Shot rows of the same Table 6 runs (values identical to the Zero-Shot rows) |
| Ablation tables (no-decomposition / no-classification / no-routing; granularity K) | `Ablation and Analysis experiment.py`; `results/ablation_experiment_results.csv` |
| Threshold sweep (`tau_low`/`tau_high`), judge rounds, token length, cross-domain | `Ablation and Analysis experiment.py`; `results/analysis_experiment_results.csv` |
| Figure 3 (cross-task ACC), Figure 4 (cross-task EM) | `画图代码1.py`, `画图代码3.py` |
| Figure 6 (threshold sweep) | `画图4.py` |
| Figure 7 (cross-domain generalization) | `画图5.py` |
| Error-distribution / repair / cost-depth figures | `新增分析实验1.py`, `新增分析实验2.py`, `新增分析实验3.py` |

Figures 3 and 4 plot the raw pre-repair Zero-Shot values; intermediate K points in the granularity figure (K = 1, 2, 4) are linearly interpolated trend indicators — only K = 0 and K = 3 were executed on the 200-task ablation set.

---

## Main results

- **Per-subtask diagnosis:** 81.25% agreement with human gold labels across the three domains.
- **Task completion:** 70.0% under binary feedback → 88.5% with the full pipeline.
- **Repair effectiveness:** 81.5% overall repair success at 1.6x baseline cost; All-Route (route every failure to GPT-4) reaches only 73.4% at 3.2x cost; All-Prompt 47.8% at 1.0x; Random 60.6% at 1.8x.
- **Strict metrics:** on DS-1000 with Kimi-k2.6, ASK-StreamDiag attains F1 = 0.900 and MCC = 0.850.
- **Cross-domain transfer:** 6–9% in-domain to cross-domain gap; accuracy stays above 73% on every transfer pair.
- **Best decomposition granularity:** three atomic subtasks; two judge rounds capture most of the gain; optimal thresholds `tau_low = 0.2`, `tau_high = 0.8` (80.9% classification accuracy).

---

## Hyperparameters

| Component | Setting |
|---|---|
| Memory size `M` | 16 |
| Prompt perturbation variants `P` | 4 (configurable via `--p-variants`) |
| MSU threshold `tau_low` | 0.2 |
| PSK threshold `tau_high` | 0.8 |
| Max judge dialogue rounds | 3 |
| Best decomposition granularity | 3 subtasks |
| Judge LLM | GPT-4, temperature 0.2 |
| Dispatcher LLM | GPT-3.5-turbo, temperature 0.0 |
| Target models | Qwen-turbo, Kimi-k2.6, Zhipu GLM-4, DouBao-seed-2.0-lite, GPT-3.5-turbo, Gemini-2.5-flash |

---

## Reproducibility notes

1. **Data path.** `common/data_loader.py` and the root scripts currently contain an absolute `DATA_BASE`/`DATASET_PATH` constant pointing at the local data directory. Change it to your own `data/` folder before running.
2. **Sample counts.** The paper reports 1,000 / 1,200 / 800 evaluated instances for DS-1000 / DDXPlus / ToolBench; the public parquet releases contain 1,000 / 1,764 / 750 rows respectively. The loader caps each dataset with `--limit` (DDXPlus is truncated to the first 1,000; all 750 ToolBench rows are used). Set the sample size explicitly when reproducing a specific table.
3. **MAM retrieval.** The original MAM-StreamICL uses FAISS + `BAAI/bge-base-en-v1.5`; the `gemini25_flash_rerun/` re-implementation uses SequenceMatcher string-similarity retrieval (top-k = 16, only correct exemplars stored). Core logic is identical; the retriever is the only difference.
4. **Dry-run values are not paper values.** `test_dry_run.py` uses mock annotations and never contacts an API; its output must not be used for reported results.
5. **No hard-coded keys.** Every script reads credentials from environment variables; no real network request is made unless the corresponding key is set.
6. **Budget controls.** The `Binary+GPT4-Routing` and `Oracle-Routing` rows of Table 7 are formula-derived (`routing = zs + (1-zs) * 0.734`, `oracle = zs + (1-zs) * 0.815`); `Binary+GPT4-Judge` is a real control run rebuilt from the paper's description.

---

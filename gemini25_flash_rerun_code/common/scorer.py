"""
Unified scoring module.
Faithfully replicates original script scoring conventions:

- ACC (Post-repair Accuracy): SequenceMatcher ratio > 0.7 threshold
  - code type: upper() + whitespace collapse before similarity
  - text type: strip().lower() before similarity
  - json type: extract api_name from response, compare against gold api_name
- EM (Exact Match): normalized string equality
- F1: f1_score([1]*n, correct_bools, average='binary', zero_division=0)
  (special writing: y_true = all ones, because ground truth is "all correct")
- MCC: matthews_corrcoef([1]*n, correct_bools)
"""
import re
import json
from difflib import SequenceMatcher
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef


# ===================== Text normalization =====================
def normalize_code(text: str) -> str:
    """Normalize code/SQL: upper, collapse whitespace."""
    if not text:
        return ""
    text = str(text).strip()
    text = text.upper()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text(text: str) -> str:
    """Normalize diagnosis text: strip + lower."""
    if not text:
        return ""
    return str(text).strip().lower()


# ===================== Answer extraction =====================
def extract_code_from_response(response: str) -> str:
    """
    Extract code from model response.
    Prefer ```sql ... ``` or ```python ... ``` or ``` code blocks.
    Fallback: try SELECT/def/class/import patterns.
    """
    if not response:
        return ""
    patterns = [
        r"```sql\s*(.*?)\s*```",
        r"```python\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"(SELECT\s+.*?;?)",
        r"(INSERT\s+.*?;?)",
        r"(UPDATE\s+.*?;?)",
        r"(DELETE\s+.*?;?)",
        r"(def\s+\w+\(.*?\).*?)$",  # Python function def for DS-1000
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE | re.DOTALL)
        if m:
            code = m.group(1).strip()
            if code:
                return code
    return response.strip()


def extract_api_name_from_response(response: str) -> str:
    """
    Extract API name from ToolBench model response.
    Look for patterns like: API_name: xxx, api_name: xxx, or JSON with 'name'/'api_name' field.
    """
    if not response:
        return ""
    # Try JSON parsing first
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for key in ["api_name", "name", "tool_name", "function_name"]:
                if key in data:
                    return str(data[key]).strip()
        except (json.JSONDecodeError, TypeError):
            pass
    # Try explicit "API_name: xxx" pattern
    m = re.search(r"API[_ ]?name\s*[:：]\s*(\w+)", response, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: first word that looks like an API name (snake_case)
    m = re.search(r"\b([a-z]+_[a-z_]+)\b", response)
    if m:
        return m.group(1).strip()
    return response.strip()


# ===================== Per-sample scoring =====================
def score_sample(pred_raw: str, gold: str, eval_type: str,
                 gold_api_input: str = None) -> Dict:
    """
    Score a single prediction against gold.
    Returns dict with: extracted_answer, valid, acc_correct, em_correct
    """
    result = {
        "extracted_answer": "",
        "valid": False,
        "acc_correct": False,
        "em_correct": False,
    }

    if not pred_raw:
        return result

    if eval_type == "code":
        extracted = extract_code_from_response(pred_raw)
        result["extracted_answer"] = extracted
        if not extracted or len(extracted) < 3:
            return result
        result["valid"] = True
        pred_norm = normalize_code(extracted)
        gold_norm = normalize_code(gold)
        sim = SequenceMatcher(None, pred_norm, gold_norm).ratio()
        result["acc_correct"] = sim > 0.7
        result["em_correct"] = (pred_norm == gold_norm)

    elif eval_type == "text":
        result["extracted_answer"] = pred_raw.strip()
        result["valid"] = len(pred_raw.strip()) > 0
        pred_norm = normalize_text(pred_raw)
        gold_norm = normalize_text(gold)
        sim = SequenceMatcher(None, pred_norm, gold_norm).ratio()
        result["acc_correct"] = sim > 0.7
        result["em_correct"] = (pred_norm == gold_norm)

    elif eval_type == "json":
        extracted_api = extract_api_name_from_response(pred_raw)
        result["extracted_answer"] = extracted_api
        if not extracted_api:
            return result
        result["valid"] = True
        # For json type, ACC = exact api name match (binary)
        pred_norm = extracted_api.strip().lower()
        gold_norm = gold.strip().lower()
        result["acc_correct"] = (pred_norm == gold_norm)
        result["em_correct"] = (pred_norm == gold_norm)

    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")

    return result


# ===================== Batch metrics =====================
def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Given list of per-sample result dicts with acc_correct / em_correct,
    compute ACC, EM, F1, MCC.
    F1 and MCC use the original script's special convention:
      y_true = [1] * n (all should be correct), y_pred = correctness booleans.
    """
    n = len(results)
    if n == 0:
        return {"ACC": 0.0, "EM": 0.0, "F1": 0.0, "MCC": 0.0}

    acc_bools = [r["acc_correct"] for r in results]
    em_bools = [r["em_correct"] for r in results]

    acc = float(np.mean(acc_bools))
    em = float(np.mean(em_bools))

    # Original special writing: y_true all 1s
    y_true = [1] * n
    f1 = f1_score(y_true, acc_bools, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_true, acc_bools)

    return {
        "ACC": round(acc, 4),
        "EM": round(em, 4),
        "F1": round(float(f1), 4),
        "MCC": round(float(mcc), 4),
    }

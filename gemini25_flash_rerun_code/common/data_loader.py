"""
Unified data loading for 3 datasets.
Auto-detects test parquet files (handles ' (1)' suffix).
Prints schema and row count at startup.

Field mapping (verified against actual parquet schemas):
- ds_1000:  question=prompt, answer=reference_code,  eval_type=code
- ddxplus:  question=PATIENT_PROFILE, answer=PATHOLOGY, eval_type=text
- toolbench: question=query, answer=api_name + api_input, eval_type=json
"""
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Base data directory — relative to the original streambench project
DATA_BASE = Path(r"C:\Users\liuxi\Desktop\streambench\data")

DATASET_CONFIG = {
    "ds_1000": {
        "dir": "ds_1000",
        "question_field": "prompt",
        "answer_field": "reference_code",
        "eval_type": "code",
        "gold_secondary": None,  # no secondary gold field
    },
    "ddxplus": {
        "dir": "ddxplus",
        "question_field": "PATIENT_PROFILE",
        "answer_field": "PATHOLOGY",
        "eval_type": "text",
        "gold_secondary": None,
    },
    "toolbench": {
        "dir": "toolbench",
        "question_field": "query",
        "answer_field": "api_name",
        "eval_type": "json",
        "gold_secondary": "api_input",  # also store api_input for full gold
    },
}


def _find_test_parquet(dataset_dir: Path) -> Path:
    """Find the test parquet file, handling ' (1)' suffix variants."""
    candidates = [
        dataset_dir / "test-00000-of-00001.parquet",
        dataset_dir / "test-00000-of-00001 (1).parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: glob any test parquet
    matches = list(dataset_dir.glob("test-*.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No test parquet found in {dataset_dir}")


def load_dataset(dataset_name: str, limit: int = 1000) -> Tuple[List[Dict], str]:
    """
    Load a dataset and return (samples, eval_type).
    Each sample: {"question": str, "answer": str, "gold_api_name": str|None, "gold_api_input": str|None}
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIG.keys())}")

    cfg = DATASET_CONFIG[dataset_name]
    ds_dir = DATA_BASE / cfg["dir"]
    parquet_path = _find_test_parquet(ds_dir)

    df = pd.read_parquet(parquet_path)
    print(f"[DataLoader] {dataset_name}: {parquet_path.name}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    samples = []
    for _, row in df.iterrows():
        q = str(row[cfg["question_field"]])
        ans = str(row[cfg["answer_field"]])

        sample = {
            "question": q,
            "answer": ans,
            "eval_type": cfg["eval_type"],
        }

        # For toolbench, also capture api_input as secondary gold
        if cfg["gold_secondary"] and cfg["gold_secondary"] in row.index:
            sample["gold_api_input"] = str(row[cfg["gold_secondary"]])

        samples.append(sample)
        if len(samples) >= limit:
            break

    print(f"  Loaded {len(samples)} samples (limit={limit})")
    return samples, cfg["eval_type"]


def get_dataset_names() -> List[str]:
    return list(DATASET_CONFIG.keys())

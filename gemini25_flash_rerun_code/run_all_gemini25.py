import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_zero_shot import run_zero_shot
from run_few_shot import run_few_shot
from run_cot import run_cot
from run_growprompt import run_growprompt
from run_self_streamicl import run_self_streamicl
from run_mam_streamicl import run_mam_streamicl
from run_evo_memory import run_evo_memory
from run_lifelong_agent_bench import run_lifelong_agent_bench
from ask_streamdiag_gemini import run_ask_streamdiag

DATASETS = ["ds_1000", "ddxplus", "toolbench"]

METHOD_ORDER = [
    ("Evo Memory", run_evo_memory),
    ("LifelongAgentBench", run_lifelong_agent_bench),
    ("Zero Shot", run_zero_shot),
    ("Few Shot", run_few_shot),
    ("CoT", run_cot),
    ("GrowPrompt", run_growprompt),
    ("MAM StreamICL", run_mam_streamicl),
    ("ASK-StreamDiag", run_ask_streamdiag),
]

OPTIONAL_METHODS = [
    ("Self-StreamICL", run_self_streamicl),
]


def run_all(output_dir: str, limit: int, include_self_streamicl: bool = False,
            dry_run: bool = False):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_methods = METHOD_ORDER + (OPTIONAL_METHODS if include_self_streamicl else [])

    results_table = {}

    for method_name, method_fn in all_methods:
        results_table[method_name] = {}
        for ds in DATASETS:
            print(f"\n{'='*60}")
            print(f"Running: {method_name} @ {ds}")
            print(f"{'='*60}")
            try:
                metrics = method_fn(ds, limit, output_dir, dry_run=dry_run)
                results_table[method_name][ds] = metrics
            except Exception as e:
                print(f"[ERROR] {method_name} @ {ds}: {e}")
                results_table[method_name][ds] = {"ACC": 0.0, "EM": 0.0, "F1": 0.0, "MCC": 0.0}

    csv_path = out_dir / "table6_gemini25_flash.csv"
    header = [
        "Method",
        "DS1000_ACC", "DDXPlus_ACC", "ToolBench_ACC",
        "DS1000_EM", "DDXPlus_EM", "ToolBench_EM",
        "DS1000_F1", "DDXPlus_F1", "ToolBench_F1",
        "DS1000_MCC", "DDXPlus_MCC", "ToolBench_MCC",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for method_name, _ in all_methods:
            row = [method_name]
            for ds in DATASETS:
                m = results_table[method_name].get(ds, {})
                row.append(m.get("ACC", 0.0))
            for ds in DATASETS:
                m = results_table[method_name].get(ds, {})
                row.append(m.get("EM", 0.0))
            for ds in DATASETS:
                m = results_table[method_name].get(ds, {})
                row.append(m.get("F1", 0.0))
            for ds in DATASETS:
                m = results_table[method_name].get(ds, {})
                row.append(m.get("MCC", 0.0))
            writer.writerow(row)

    print(f"\nSummary CSV written to: {csv_path}")
    print("\nFinal Table:")
    print(" | ".join(header))
    for method_name, _ in all_methods:
        row = [method_name]
        for ds in DATASETS:
            m = results_table[method_name].get(ds, {})
            row.append(f"{m.get('ACC', 0.0):.4f}")
        for ds in DATASETS:
            m = results_table[method_name].get(ds, {})
            row.append(f"{m.get('EM', 0.0):.4f}")
        for ds in DATASETS:
            m = results_table[method_name].get(ds, {})
            row.append(f"{m.get('F1', 0.0):.4f}")
        for ds in DATASETS:
            m = results_table[method_name].get(ds, {})
            row.append(f"{m.get('MCC', 0.0):.4f}")
        print(" | ".join(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--include-self-streamicl", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_all(args.output_dir, args.limit, args.include_self_streamicl, args.dry_run)

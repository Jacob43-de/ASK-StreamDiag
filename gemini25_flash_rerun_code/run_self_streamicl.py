
import argparse
import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common.gemini_client import GeminiClient
from common.data_loader import load_dataset
from common.scorer import score_sample, compute_metrics
from common.prompts import build_memory_prompt

MAX_MEMORY = 16
SIMILARITY_THRESHOLD = 0.9


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def memory_evolve(memory_list: list, new_item: dict) -> bool:
    """Add new item if not too similar to existing. Sort by similarity to new, cap at MAX_MEMORY."""
    for item in memory_list:
        if similarity(item["q"], new_item["q"]) > SIMILARITY_THRESHOLD:
            return False
    memory_list.append(new_item)
    memory_list.sort(key=lambda x: similarity(x["q"], new_item["q"]), reverse=True)
    if len(memory_list) > MAX_MEMORY:
        memory_list.pop(-1)
    return True


def run_self_streamicl(dataset_name: str, limit: int, output_dir: str,
                       dry_run: bool = False, dry_run_client=None):
    samples, eval_type = load_dataset(dataset_name, limit=limit)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_outputs/self_streamicl_{dataset_name}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    done_indices = set()
    if raw_path.exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    done_indices.add(rec["index"])
                except Exception:
                    pass
    print(f"[Self-StreamICL] Already done: {len(done_indices)} samples, resuming.")

    if dry_run:
        client = dry_run_client or (lambda p: f"MOCK_RESPONSE for: {p[:50]}...")
    else:
        client = GeminiClient(temperature=0.0)

    memory = []

    with open(raw_path, "a", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            if idx in done_indices:
                continue
            question = sample["question"]
            gold = sample["answer"]

            prompt = build_memory_prompt(dataset_name, question, memory)
            t0 = time.time()
            response = client(prompt)
            elapsed = time.time() - t0

            scored = score_sample(response, gold, eval_type,
                                  gold_api_input=sample.get("gold_api_input"))

            # Memory evolve: only store correct answers
            if scored["acc_correct"]:
                memory_evolve(memory, {"q": question, "a": scored["extracted_answer"]})

            record = {
                "index": idx,
                "method": "self_streamicl",
                "dataset": dataset_name,
                "question": question,
                "gold": gold,
                "raw_response": response,
                "extracted_answer": scored["extracted_answer"],
                "valid": scored["valid"],
                "acc_correct": scored["acc_correct"],
                "em_correct": scored["em_correct"],
                "memory_size": len(memory),
                "elapsed_sec": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if (idx + 1) % 10 == 0:
                print(f"[Self-StreamICL][{dataset_name}] {idx+1}/{len(samples)} done, mem={len(memory)}")

    all_results = []
    if raw_path.exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    all_results.append(rec)
                except Exception:
                    pass

    metrics = compute_metrics(all_results)
    print(f"[Self-StreamICL][{dataset_name}] Final: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ds_1000", "ddxplus", "toolbench"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_self_streamicl(args.dataset, args.limit, args.output_dir, dry_run=args.dry_run)

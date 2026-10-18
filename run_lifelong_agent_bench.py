import argparse
import json
import random
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common.gemini_client import GeminiClient
from common.data_loader import load_dataset
from common.scorer import score_sample, compute_metrics

NUM_SELF_CONSISTENCY = 5
GROUP_SIZE = 3
SHORT_TERM_MEMORY_MAX = 5


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def group_self_consistency(client, question: str, num_samples: int, group_size: int) -> str:
    answers = []
    for _ in range(num_samples):
        prompt = f"请回答以下问题，可以适当发挥创造力：\n{question}"
        ans = client(prompt)
        answers.append(ans)

    random.shuffle(answers)
    groups = [answers[i:i+group_size] for i in range(0, len(answers), group_size)]
    group_votes = []
    for group in groups:
        if not group:
            continue
        best = max(set(group), key=lambda x: sum(similarity(x, other) for other in group))
        group_votes.append(best)

    if not group_votes:
        return ""
    final = max(set(group_votes), key=lambda x: group_votes.count(x))
    return final


def run_lifelong_agent_bench(dataset_name: str, limit: int, output_dir: str,
                             dry_run: bool = False, dry_run_client=None):
    samples, eval_type = load_dataset(dataset_name, limit=limit)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_outputs/lifelong_agent_bench_{dataset_name}.jsonl"
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
    print(f"[LifelongAgentBench] Already done: {len(done_indices)} samples, resuming.")

    if dry_run:
        client = dry_run_client or (lambda p: f"MOCK_RESPONSE for: {p[:50]}...")
    else:
        # Note: self-consistency needs higher temperature for diversity.
        # Original uses temperature perturbation; Gemini 2.5 Flash at 0.0
        # will produce nearly identical outputs. We use temperature=0.3 for sampling.
        client = GeminiClient(temperature=0.3)

    short_term_memory = []

    with open(raw_path, "a", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            if idx in done_indices:
                continue
            question = sample["question"]
            gold = sample["answer"]

            t0 = time.time()
            response = group_self_consistency(client, question, NUM_SELF_CONSISTENCY, GROUP_SIZE)
            elapsed = time.time() - t0

            scored = score_sample(response, gold, eval_type,
                                  gold_api_input=sample.get("gold_api_input"))

            if scored["acc_correct"]:
                short_term_memory.append({"question": question, "answer": scored["extracted_answer"]})
                if len(short_term_memory) > SHORT_TERM_MEMORY_MAX:
                    short_term_memory.pop(0)

            record = {
                "index": idx,
                "method": "lifelong_agent_bench",
                "dataset": dataset_name,
                "question": question,
                "gold": gold,
                "raw_response": response,
                "extracted_answer": scored["extracted_answer"],
                "valid": scored["valid"],
                "acc_correct": scored["acc_correct"],
                "em_correct": scored["em_correct"],
                "memory_size": len(short_term_memory),
                "elapsed_sec": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if (idx + 1) % 10 == 0:
                print(f"[LifelongAgentBench][{dataset_name}] {idx+1}/{len(samples)} done")

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
    print(f"[LifelongAgentBench][{dataset_name}] Final: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ds_1000", "ddxplus", "toolbench"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_lifelong_agent_bench(args.dataset, args.limit, args.output_dir, dry_run=args.dry_run)

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
from common.prompts import build_few_shot_prompt

MAX_MEMORY_SIZE = 1000
RETRIEVAL_K = 16


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def retrieve_topk(memory: list, query: str, k: int) -> list:
    """Retrieve top-k most similar examples from memory using string similarity."""
    if not memory:
        return []
    scored = [(mem, similarity(mem["q"], query)) for mem in memory]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [mem for mem, _ in scored[:k]]


def run_mam_streamicl(dataset_name: str, limit: int, output_dir: str,
                      dry_run: bool = False, dry_run_client=None):
    samples, eval_type = load_dataset(dataset_name, limit=limit)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_outputs/mam_streamicl_{dataset_name}.jsonl"
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
    print(f"[MAM-StreamICL] Already done: {len(done_indices)} samples, resuming.")

    if dry_run:
        client = dry_run_client or (lambda p: f"MOCK_RESPONSE for: {p[:50]}...")
    else:
        client = GeminiClient(temperature=0.0)

    memory = []  # shared memory: list of {"q": str, "a": str}

    with open(raw_path, "a", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            if idx in done_indices:
                continue
            question = sample["question"]
            gold = sample["answer"]

            # Retrieve top-k similar examples
            retrieved = retrieve_topk(memory, question, k=RETRIEVAL_K)

            # Build few-shot prompt with retrieved examples
            # Adapt: use retrieved examples inline
            example_text = ""
            for mem in retrieved:
                example_text += f"用户问题：{mem['q']}\n答案：{mem['a']}\n\n"

            if dataset_name == "ds_1000":
                task = "Python代码"
            elif dataset_name == "ddxplus":
                task = "诊断结果"
            else:
                task = "API调用"

            prompt = f"""以下是参考示例：
{example_text}
请根据用户问题生成正确的{task}，只输出{task}，不要解释。
用户问题：{question}
{task}："""

            t0 = time.time()
            response = client(prompt)
            elapsed = time.time() - t0

            scored = score_sample(response, gold, eval_type,
                                  gold_api_input=sample.get("gold_api_input"))

            # Update memory: only store correct answers
            if scored["acc_correct"]:
                memory.append({"q": question, "a": scored["extracted_answer"]})
                if len(memory) > MAX_MEMORY_SIZE:
                    memory.pop(0)

            record = {
                "index": idx,
                "method": "mam_streamicl",
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
                print(f"[MAM-StreamICL][{dataset_name}] {idx+1}/{len(samples)} done, mem={len(memory)}")

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
    print(f"[MAM-StreamICL][{dataset_name}] Final: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ds_1000", "ddxplus", "toolbench"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_mam_streamicl(args.dataset, args.limit, args.output_dir, dry_run=args.dry_run)


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
MEMORY_RETRIEVAL_TOPK = 3


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def retrieve_memories(memory_list: list, query: str, k: int) -> list:
    if not memory_list:
        return []
    scored = [(mem, similarity(mem["question"], query)) for mem in memory_list]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [mem for mem, _ in scored[:k]]


def evolve_memory(memory_list: list, new_item: dict, max_size: int) -> list:
    for i, mem in enumerate(memory_list):
        if similarity(mem["question"], new_item["question"]) > 0.9:
            memory_list[i] = new_item
            return memory_list
    memory_list.append(new_item)
    memory_list.sort(key=lambda x: similarity(x["question"], new_item["question"]), reverse=True)
    if len(memory_list) > max_size:
        memory_list.pop()
    return memory_list


def run_evo_memory(dataset_name: str, limit: int, output_dir: str,
                   dry_run: bool = False, dry_run_client=None):
    samples, eval_type = load_dataset(dataset_name, limit=limit)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_outputs/evo_memory_{dataset_name}.jsonl"
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
    print(f"[Evo Memory] Already done: {len(done_indices)} samples, resuming.")

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

            retrieved = retrieve_memories(memory, question, MEMORY_RETRIEVAL_TOPK)
            # Build prompt with retrieved memories
            mem_text = "\n".join([f"问题：{m['question']}\n答案：{m['answer']}" for m in retrieved])

            if dataset_name == "ds_1000":
                task = "Python代码"
            elif dataset_name == "ddxplus":
                task = "诊断结果"
            else:
                task = "API调用"

            prompt = f"""你是一个专业的AI助手。请根据以下参考示例，回答用户问题。
参考示例：
{mem_text if mem_text else "无"}
用户问题：{question}
请直接输出{task}，不要解释。"""

            t0 = time.time()
            response = client(prompt)
            elapsed = time.time() - t0

            scored = score_sample(response, gold, eval_type,
                                  gold_api_input=sample.get("gold_api_input"))

            if scored["acc_correct"]:
                evolve_memory(memory, {"question": question, "answer": scored["extracted_answer"]}, MAX_MEMORY)

            record = {
                "index": idx,
                "method": "evo_memory",
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
                print(f"[Evo Memory][{dataset_name}] {idx+1}/{len(samples)} done, mem={len(memory)}")

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
    print(f"[Evo Memory][{dataset_name}] Final: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ds_1000", "ddxplus", "toolbench"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_evo_memory(args.dataset, args.limit, args.output_dir, dry_run=args.dry_run)

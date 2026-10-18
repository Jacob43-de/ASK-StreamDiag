"""
ASK-StreamDiag main pipeline with Gemini 2.5 Flash as target model.
Corresponds to: "Ablation and Analysis experiment.py" + "Self-StreamICL（Gemini）.py"
Paper §5.4: dispatcher=GPT-3.5-turbo, judge=GPT-4 (temp=0.2), target=Gemini 2.5 Flash.

Pipeline per sample:
1. GPT-3.5-turbo decomposes question into subtasks
2. Gemini 2.5 Flash generates answer (temperature=0.0)
3. GPT-4 judges each subtask (success/failure)
4. For failed subtasks: prompt perturbation P times → re-run on target → re-judge
5. MSU/PSK classification based on success rate
6. Real routing repair:
   - MSU → GPT-4 directly answers the subtask
   - PSK → prompt rephrase, retry on target model
7. Post-repair overall correctness evaluation
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from common.gemini_client import GeminiClient
from common.openai_clients import gpt4_judge, gpt35_dispatcher, call_openai_chat
from common.data_loader import load_dataset
from common.scorer import score_sample, compute_metrics, normalize_code, normalize_text
from common.prompts import (
    SUBTASKS_BY_DATASET,
    build_judge_prompt,
    build_decomposition_prompt,
    build_rephrase_prompt,
    build_add_example_suffix,
)

# Hyperparameters (paper §5.4)
MAX_MEMORY = 16
P_NUM_VARIANTS = 4          # Paper says 3-4; ablation script uses 5. We default to 4.
TAU_LOW = 0.2
TAU_HIGH = 0.8
ROUTING_UPDATE_INTERVAL = 50


def judge_subtask(dataset_name: str, question: str, generated_answer: str,
                  subtask_desc: str) -> Tuple[bool, str]:
    """Call GPT-4 to judge if a subtask is satisfied."""
    prompt = build_judge_prompt(dataset_name, question, generated_answer, subtask_desc)
    response = gpt4_judge(prompt)
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return result.get("success", False), result.get("reason", "")
        except (json.JSONDecodeError, TypeError):
            return False, "JSON parse failed"
    return False, "Judge output format error"


def classify_msu_psk(dataset_name: str, question: str, subtask_desc: str,
                     target_client, p_variants: int = P_NUM_VARIANTS) -> Tuple[str, float]:
    """
    Prompt perturbation test: rephrase question P times, ask target model,
    re-judge the subtask. Return (classification, success_rate).
    """
    success_count = 0
    variant_types = ["rephrase", "add_example"]

    for i in range(p_variants):
        vtype = variant_types[i % len(variant_types)]
        if vtype == "rephrase":
            rephrase_prompt = build_rephrase_prompt(question)
            variant_q = gpt35_dispatcher(rephrase_prompt)
            if not variant_q:
                variant_q = question
        else:  # add_example
            suffix = build_add_example_suffix(dataset_name)
            variant_q = question + suffix

        # Ask target model with variant question
        if dataset_name == "ds_1000":
            task = "Python代码"
        elif dataset_name == "ddxplus":
            task = "诊断结果"
        else:
            task = "API调用"
        test_prompt = f"你是{task}专家，根据用户问题生成正确的{task}。\n用户问题：{variant_q}\n只输出{task}："
        new_answer = target_client(test_prompt)
        if not new_answer:
            continue

        success, _ = judge_subtask(dataset_name, variant_q, new_answer, subtask_desc)
        if success:
            success_count += 1

    r = success_count / p_variants if p_variants > 0 else 0.0
    if r < TAU_LOW:
        return "MSU", r
    elif r < TAU_HIGH:
        return "PSK", r
    else:
        return "Reliable", r


def route_repair_msu(dataset_name: str, question: str, subtask_desc: str) -> str:
    """MSU: GPT-4 directly answers the subtask."""
    if dataset_name == "ds_1000":
        task = "Python代码"
    elif dataset_name == "ddxplus":
        task = "诊断结果"
    else:
        task = "API调用"

    prompt = f"""你是一个资深{task}专家。请解决以下问题中指定的子任务。
用户问题：{question}
需要解决的子任务：{subtask_desc}
请直接给出该子任务的答案："""
    return call_openai_chat(
        model="gpt-4",
        system_prompt=f"你是一个资深{task}专家。请直接给出答案。",
        user_prompt=prompt,
        temperature=0.2,
    )


def route_repair_psk(dataset_name: str, question: str, subtask_desc: str,
                     target_client) -> str:
    """PSK: rephrase prompt, retry on target model."""
    rephrase_prompt = build_rephrase_prompt(question)
    rephrased = gpt35_dispatcher(rephrase_prompt)
    if not rephrased:
        rephrased = question

    if dataset_name == "ds_1000":
        task = "Python代码"
    elif dataset_name == "ddxplus":
        task = "诊断结果"
    else:
        task = "API调用"

    retry_prompt = f"""请仔细思考并解决以下问题，特别注意：{subtask_desc}
用户问题：{rephrased}
只输出{task}："""
    return target_client(retry_prompt)


def run_ask_streamdiag(dataset_name: str, limit: int, output_dir: str,
                       p_variants: int = P_NUM_VARIANTS,
                       dry_run: bool = False,
                       dry_run_target_client=None,
                       dry_run_judge_fn=None,
                       dry_run_dispatcher_fn=None):
    samples, eval_type = load_dataset(dataset_name, limit=limit)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_outputs/ask_streamdiag_{dataset_name}.jsonl"
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
    print(f"[ASK-StreamDiag] Already done: {len(done_indices)} samples, resuming.")

    # Initialize clients
    if dry_run:
        target_client = dry_run_target_client or (lambda p: f"MOCK_TARGET for: {p[:50]}...")
        judge_fn = dry_run_judge_fn or (lambda q, a, d: (True, "mock"))
        dispatcher_fn = dry_run_dispatcher_fn or (lambda p: "mock decomposition")
    else:
        target_client = GeminiClient(temperature=0.0)
        judge_fn = lambda q, a, d: judge_subtask(dataset_name, q, a, d)
        dispatcher_fn = gpt35_dispatcher

    subtask_defs = SUBTASKS_BY_DATASET[dataset_name]
    memory = []  # M=16 memory of correct answers

    with open(raw_path, "a", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            if idx in done_indices:
                continue
            question = sample["question"]
            gold = sample["answer"]

            # Step 1: Subtask decomposition (GPT-3.5-turbo dispatcher)
            decomp_prompt = build_decomposition_prompt(dataset_name, question)
            decomposition = dispatcher_fn(decomp_prompt)

            # Step 2: Target model (Gemini 2.5 Flash) generates answer
            mem_text = ""
            for m in memory[-MAX_MEMORY:]:
                mem_text += f"问题：{m['q']}\n答案：{m['a']}\n"

            if dataset_name == "ds_1000":
                task = "Python代码"
            elif dataset_name == "ddxplus":
                task = "诊断结果"
            else:
                task = "API调用"

            initial_prompt = f"""你是{task}专家。根据用户问题生成正确的{task}。
参考示例：
{mem_text if mem_text else "无"}
用户问题：{question}
只输出{task}，不要解释。"""
            t0 = time.time()
            initial_answer = target_client(initial_prompt)
            elapsed = time.time() - t0

            # Step 3: GPT-4 judges each subtask
            subtask_results = []
            for sub in subtask_defs:
                success, reason = judge_fn(question, initial_answer, sub["desc"])
                subtask_results.append({
                    "name": sub["name"],
                    "success": success,
                    "reason": reason,
                })

            # Step 4: Score initial answer
            initial_scored = score_sample(initial_answer, gold, eval_type,
                                          gold_api_input=sample.get("gold_api_input"))
            post_repair_answer = initial_answer
            post_repair_correct = initial_scored["acc_correct"]
            repair_applied = "none"
            msu_psk_log = []

            # Step 5: If overall incorrect, diagnose and repair
            if not post_repair_correct:
                failed_subtasks = [s for s in subtask_results if not s["success"]]
                for fs in failed_subtasks:
                    # Classify MSU/PSK via prompt perturbation
                    msu_psk_type, success_rate = classify_msu_psk(
                        dataset_name, question, fs["desc"], target_client, p_variants
                    )
                    msu_psk_log.append({
                        "subtask": fs["name"],
                        "type": msu_psk_type,
                        "success_rate": round(success_rate, 3),
                    })

                    if msu_psk_type == "MSU":
                        # MSU: GPT-4 directly answers
                        repaired = route_repair_msu(dataset_name, question, fs["desc"])
                        if repaired:
                            post_repair_answer = repaired
                            repair_applied = "MSU_routed_to_GPT4"
                    elif msu_psk_type == "PSK":
                        # PSK: rephrase + retry on target
                        retried = route_repair_psk(dataset_name, question, fs["desc"], target_client)
                        if retried:
                            post_repair_answer = retried
                            repair_applied = "PSK_reprompted_target"

                # Re-score after repair
                post_repair_scored = score_sample(post_repair_answer, gold, eval_type,
                                                  gold_api_input=sample.get("gold_api_input"))
                post_repair_correct = post_repair_scored["acc_correct"]
                post_repair_em = post_repair_scored["em_correct"]
                post_repair_extracted = post_repair_scored["extracted_answer"]
            else:
                post_repair_em = initial_scored["em_correct"]
                post_repair_extracted = initial_scored["extracted_answer"]

            # Update memory with post-repair correct answers (sliding window, M=16)
            if post_repair_correct:
                memory.append({"q": question, "a": post_repair_extracted})
                memory = memory[-MAX_MEMORY:]

            record = {
                "index": idx,
                "method": "ask_streamdiag",
                "dataset": dataset_name,
                "question": question,
                "gold": gold,
                "decomposition": decomposition,
                "initial_response": initial_answer,
                "post_repair_response": post_repair_answer,
                "extracted_answer": post_repair_extracted,
                "valid": post_repair_correct or initial_scored["valid"],
                "acc_correct": post_repair_correct,
                "em_correct": post_repair_em,
                "subtask_results": subtask_results,
                "msu_psk_log": msu_psk_log,
                "repair_applied": repair_applied,
                "elapsed_sec": round(elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if (idx + 1) % 10 == 0:
                print(f"[ASK-StreamDiag][{dataset_name}] {idx+1}/{len(samples)} done, repair={repair_applied}")

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
    print(f"[ASK-StreamDiag][{dataset_name}] Final: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ds_1000", "ddxplus", "toolbench"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--p-variants", type=int, default=P_NUM_VARIANTS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_ask_streamdiag(args.dataset, args.limit, args.output_dir,
                       p_variants=args.p_variants, dry_run=args.dry_run)

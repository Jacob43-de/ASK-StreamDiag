import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common.data_loader import load_dataset
from common.scorer import score_sample, compute_metrics, extract_code_from_response, extract_api_name_from_response


def mock_client_factory(dataset_name: str, eval_type: str):
    """Return a mock client that produces a realistic-looking response based on dataset type."""
    def _mock(prompt: str) -> str:
        if eval_type == "code":
            # Mock: return a Python code block
            return "```python\ndf[df['Col1'] > 5]\n```"
        elif eval_type == "text":
            # Mock: return a diagnosis
            return "Acute COPD exacerbation / infection"
        else:  # json
            return "API_name: add_date\n参数: {\"date\": \"2022-10-14\", \"days\": 90}"
    return _mock


def test_data_loading():
    print("=" * 60)
    print("TEST 1: Data loading from all 3 parquets")
    print("=" * 60)
    for ds in ["ds_1000", "ddxplus", "toolbench"]:
        samples, eval_type = load_dataset(ds, limit=5)
        assert len(samples) > 0, f"No samples loaded for {ds}"
        assert "question" in samples[0]
        assert "answer" in samples[0]
        print(f"  {ds}: {len(samples)} samples, eval_type={eval_type}")
        print(f"    Q[0]: {samples[0]['question'][:80]}...")
        print(f"    A[0]: {samples[0]['answer'][:80]}...")
    print("PASSED\n")


def test_scoring():
    print("=" * 60)
    print("TEST 2: Scoring")
    print("=" * 60)

    # Code type
    r = score_sample("```python\ndf[df['Col1'] > 5]\n```", "df[df['Col1'] > 5]", "code")
    print(f"  code exact: valid={r['valid']}, acc={r['acc_correct']}, em={r['em_correct']}")
    assert r["valid"] == True

    # Text type
    r = score_sample("Acute COPD exacerbation / infection", "Acute COPD exacerbation / infection", "text")
    print(f"  text exact: valid={r['valid']}, acc={r['acc_correct']}, em={r['em_correct']}")
    assert r["acc_correct"] == True

    # JSON type
    r = score_sample("API_name: add_date", "add_date", "json")
    print(f"  json exact: valid={r['valid']}, acc={r['acc_correct']}, em={r['em_correct']}")
    assert r["acc_correct"] == True

    # Batch metrics
    fake_results = [
        {"acc_correct": True, "em_correct": True},
        {"acc_correct": True, "em_correct": False},
        {"acc_correct": False, "em_correct": False},
        {"acc_correct": True, "em_correct": True},
    ]
    metrics = compute_metrics(fake_results)
    print(f"  batch metrics: {metrics}")
    assert 0.0 <= metrics["ACC"] <= 1.0
    assert 0.0 <= metrics["F1"] <= 1.0
    print("PASSED\n")


def test_extraction():
    print("=" * 60)
    print("TEST 3: Answer extraction")
    print("=" * 60)

    # Code extraction
    code = extract_code_from_response("Here is the solution:\n```python\ndf.groupby('Type').sum()\n```\nHope it helps!")
    print(f"  code extracted: {code[:60]}...")
    assert "groupby" in code

    # API name extraction
    api = extract_api_name_from_response("I will call add_date with the given parameters.")
    print(f"  api extracted: {api}")
    assert "add_date" in api
    print("PASSED\n")


def test_thought_part_filtering():
    print("=" * 60)
    print("TEST 3b: Gemini 2.5 thought part filtering")
    print("=" * 60)
    from common.gemini_client import _extract_text

    # Case 1: Response with thought part + formal answer part
    resp_with_thought = {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Let me think step by step...", "thought": True},
                    {"text": "正式答案", "thought": False},
                ]
            }
        }]
    }
    result = _extract_text(resp_with_thought)
    print(f"  with thought: '{result}'")
    assert result == "正式答案", f"Expected '正式答案', got '{result}'"

    # Case 2: Backward compatibility — no thought field at all (old 1.5 Flash structure)
    resp_no_thought = {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "普通答案 part 1"},
                    {"text": "普通答案 part 2"},
                ]
            }
        }]
    }
    result2 = _extract_text(resp_no_thought)
    print(f"  without thought (backward compat): '{result2}'")
    assert "普通答案 part 1" in result2
    assert "普通答案 part 2" in result2

    # Case 3: All parts are thought (edge case)
    resp_all_thought = {
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "思考1", "thought": True},
                    {"text": "思考2", "thought": True},
                ]
            }
        }]
    }
    result3 = _extract_text(resp_all_thought)
    print(f"  all thought parts: '{result3}'")
    assert result3 == ""

    print("PASSED\n")


def test_method_dry_run():
    print("=" * 60)
    print("TEST 4: Method dry-run (zero-shot, 2 samples per dataset)")
    print("=" * 60)
    from run_zero_shot import run_zero_shot

    with tempfile.TemporaryDirectory() as tmpdir:
        for ds in ["ds_1000", "ddxplus", "toolbench"]:
            samples, eval_type = load_dataset(ds, limit=2)
            mock_client = mock_client_factory(ds, eval_type)
            metrics = run_zero_shot(ds, limit=2, output_dir=tmpdir,
                                   dry_run=True, dry_run_client=mock_client)
            print(f"  {ds}: {metrics}")

            # Verify raw output file exists
            raw_file = Path(tmpdir) / "raw_outputs" / f"zero_shot_{ds}.jsonl"
            assert raw_file.exists(), f"Raw file not found: {raw_file}"
            lines = raw_file.read_text(encoding="utf-8").strip().split("\n")
            print(f"    raw_outputs lines: {len(lines)}")
            assert len(lines) == 2

    print("PASSED\n")


def test_resume_logic():
    print("=" * 60)
    print("TEST 5: Resume logic")
    print("=" * 60)
    from run_zero_shot import run_zero_shot

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = "ds_1000"
        samples, eval_type = load_dataset(ds, limit=3)
        mock_client = mock_client_factory(ds, eval_type)

        # First run: 3 samples
        run_zero_shot(ds, limit=3, output_dir=tmpdir, dry_run=True, dry_run_client=mock_client)
        raw_file = Path(tmpdir) / "raw_outputs" / f"zero_shot_{ds}.jsonl"
        lines1 = raw_file.read_text(encoding="utf-8").strip().split("\n")
        count1 = len(lines1)
        print(f"  After first run: {count1} records")

        # Second run: should skip all 3 (no new writes)
        run_zero_shot(ds, limit=3, output_dir=tmpdir, dry_run=True, dry_run_client=mock_client)
        lines2 = raw_file.read_text(encoding="utf-8").strip().split("\n")
        count2 = len(lines2)
        print(f"  After second run: {count2} records")
        assert count1 == count2, f"Resume failed: {count1} != {count2}"

    print("PASSED\n")


def test_csv_summary():
    print("=" * 60)
    print("TEST 6: CSV summary format (mock data)")
    print("=" * 60)
    import csv

    csv_path = Path("table6_gemini25_flash_MOCK.csv")
    header = [
        "Method",
        "DS1000_ACC", "DDXPlus_ACC", "ToolBench_ACC",
        "DS1000_EM", "DDXPlus_EM", "ToolBench_EM",
        "DS1000_F1", "DDXPlus_F1", "ToolBench_F1",
        "DS1000_MCC", "DDXPlus_MCC", "ToolBench_MCC",
    ]
    mock_rows = [
        ["Evo Memory", 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5],
        ["ASK-StreamDiag", 0.8, 0.8, 0.9, 0.6, 0.7, 0.8, 0.8, 0.8, 0.9, 0.7, 0.7, 0.8],
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(mock_rows)
    print(f"  Mock CSV written: {csv_path}")
    print(f"  Header: {header}")
    print(f"  Rows: {len(mock_rows)} (MOCK values, NOT for paper)")
    print("PASSED\n")


def test_table7_formula():
    print("=" * 60)
    print("TEST 7: Table 7 formula derivation (routing / oracle)")
    print("=" * 60)
    from run_budget_controls import compute_formula_rows

    # Known test vectors (from Gemini 1.5 Flash old Table 6 Zero-Shot row)
    zs = {"ds_1000": 0.692, "ddxplus": 0.718, "toolbench": 0.682}
    ask = {"ds_1000": 0.791, "ddxplus": 0.828, "toolbench": 0.909}  # placeholder, not asserted

    result = compute_formula_rows(zs, ask)
    routing = result["Binary+GPT4-Routing"]
    oracle = result["Oracle-Routing"]

    print(f"  zs: {zs}")
    print(f"  routing: {routing}")
    print(f"  oracle: {oracle}")

    # Expected values (±0.0005 tolerance)
    expected_routing = {"ds_1000": 0.918, "ddxplus": 0.925, "toolbench": 0.915}
    expected_oracle = {"ds_1000": 0.943, "ddxplus": 0.948, "toolbench": 0.941}

    for ds in expected_routing:
        diff = abs(routing[ds] - expected_routing[ds])
        print(f"  routing {ds}: got={routing[ds]:.3f}, expected={expected_routing[ds]:.3f}, diff={diff:.4f}")
        assert diff < 0.0005, f"Routing {ds} mismatch: {routing[ds]} vs {expected_routing[ds]}"

    for ds in expected_oracle:
        diff = abs(oracle[ds] - expected_oracle[ds])
        print(f"  oracle {ds}: got={oracle[ds]:.3f}, expected={expected_oracle[ds]:.3f}, diff={diff:.4f}")
        assert diff < 0.0005, f"Oracle {ds} mismatch: {oracle[ds]} vs {expected_oracle[ds]}"

    print("PASSED\n")


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# DRY-RUN OFFLINE TEST SUITE (no real API calls)")
    print("#" * 60 + "\n")

    test_data_loading()
    test_extraction()
    test_thought_part_filtering()
    test_scoring()
    test_method_dry_run()
    test_resume_logic()
    test_csv_summary()
    test_table7_formula()

    print("#" * 60)
    print("# ALL DRY-RUN TESTS PASSED")
    print("#" * 60)

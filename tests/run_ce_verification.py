"""
CE Correction Verification Runner
-----------------------------------
Run this script after ANY modification to:
  - core/gpt_caller.py   (system prompt, prefix guide, etc.)
  - core/fuzzy_matcher.py (matching algorithm, pre-conditions)

It sends all 15 CE-validated test cases to the API and checks whether
the AI_MATERIAL_CATEGORY matches the expected CE_MATERIAL_CATEGORY.

Usage:
  python tests/run_ce_verification.py [--server http://localhost:8000]
"""
import json, sys, time, argparse, requests
from pathlib import Path

TEST_FILE = Path(__file__).parent / "ce_correction_verification.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.server.rstrip("/")

    with open(TEST_FILE) as f:
        data = json.load(f)
    cases = data["test_cases"]
    item_numbers = [c["item_number"] for c in cases]
    expected = {c["item_number"]: c["expected_category"] for c in cases}

    print(f"=== CE Correction Verification ({len(cases)} test cases) ===")
    print(f"Server: {base}\n")

    # Submit lookup job
    resp = requests.post(f"{base}/api/lookup/run", json={"item_numbers": item_numbers, "vector_top_k": 10})
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]
    print(f"Job submitted: {job_id}")

    # Poll until done
    for _ in range(120):
        time.sleep(5)
        status_resp = requests.get(f"{base}/api/batch/{job_id}/status")
        status_resp.raise_for_status()
        status = status_resp.json()
        if status["status"] in ("done", "error"):
            break
        print(f"  ... {status.get('status', '?')} ({status.get('high', 0)}H/{status.get('medium', 0)}M/{status.get('low', 0)}L)")
    else:
        print("ERROR: Timed out waiting for job to complete")
        sys.exit(1)

    if status["status"] == "error":
        print(f"ERROR: Job failed — {status.get('error', '?')}")
        sys.exit(1)

    results = {r["Item_Number"]: r for r in status.get("results", [])}

    # Compare
    passed, failed = 0, 0
    print(f"\n{'='*100}")
    print(f"{'Item_Number':<18} {'Expected':<15} {'AI Result':<15} {'Conf':<8} {'Status':<6}")
    print(f"{'='*100}")

    for case in cases:
        item_no = case["item_number"]
        exp = case["expected_category"]
        r = results.get(item_no)
        if not r:
            print(f"{item_no:<18} {exp:<15} {'NOT FOUND':<15} {'—':<8} FAIL")
            failed += 1
            continue
        ai_cat = r.get("AI_MATERIAL_CATEGORY", "")
        conf = r.get("AI_confidence", "")
        ok = ai_cat == exp
        status_str = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{item_no:<18} {exp:<15} {ai_cat:<15} {conf:<8} {status_str}")

    print(f"{'='*100}")
    print(f"\nResults: {passed}/{len(cases)} passed, {failed}/{len(cases)} failed")
    pct = passed / len(cases) * 100
    print(f"Accuracy: {pct:.1f}%")

    if failed > 0:
        print("\n--- Failed Cases Detail ---")
        for case in cases:
            item_no = case["item_number"]
            exp = case["expected_category"]
            r = results.get(item_no)
            ai_cat = r.get("AI_MATERIAL_CATEGORY", "") if r else ""
            if ai_cat != exp:
                print(f"\n  {item_no}: {case['item_desc']}")
                print(f"    MPN: {case['mpn']}")
                print(f"    Expected: {exp} | Got: {ai_cat}")
                print(f"    Rule: {case['correction_rule']}")
                if r:
                    print(f"    AI Reason: {r.get('AI_reason', '')[:200]}")
        sys.exit(1)
    else:
        print("\nAll CE correction test cases passed!")

if __name__ == "__main__":
    main()

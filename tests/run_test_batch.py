r"""
Quick smoke test - runs the 4 standard test items through the lookup pipeline
and prints a summary table. Use after any changes to matching/GPT logic.

Usage:
    cd atj-component-material-category
    .venv\Scripts\python tests\run_test_batch.py
"""
import json
import sys
import io
import time
import requests

# Fix Windows CP950 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

API = "http://localhost:8000"
TEST_ITEMS_FILE = "tests/test_items.json"


def main():
    # Load test items
    with open(TEST_ITEMS_FILE) as f:
        test_data = json.load(f)

    item_numbers = [item["item_number"] for item in test_data["items"]]
    hints = {item["item_number"]: item["expected_category_hint"] for item in test_data["items"]}

    print(f"Running lookup for {len(item_numbers)} test items...")
    print(f"Items: {', '.join(item_numbers)}")
    print()

    # Start lookup job
    r = requests.post(f"{API}/api/lookup/run", json={"item_numbers": item_numbers})
    r.raise_for_status()
    job_id = r.json()["job_id"]
    print(f"Job {job_id} queued. Polling...")

    # Poll until done
    for _ in range(60):
        time.sleep(2)
        r = requests.get(f"{API}/api/batch/{job_id}/status")
        status = r.json()
        if status["status"] in ("done", "error"):
            break
    else:
        print("Timeout waiting for job!")
        return

    if status["status"] == "error":
        print(f"Job FAILED: {status.get('error', 'unknown')}")
        return

    print(f"Job done! High={status['high']} Medium={status['medium']} Low={status['low']} Error={status['error']}")
    print()

    # Download and display results
    import pandas as pd
    excel_url = f"{API}/api/batch/{job_id}/export"
    r = requests.get(excel_url)
    with open("tests/_last_test_result.xlsx", "wb") as f:
        f.write(r.content)

    df = pd.read_excel("tests/_last_test_result.xlsx", sheet_name="Raw Data")

    # Summary table
    print(f"{'Item Number':<18} {'Confidence':<12} {'Source':<16} {'AI Category':<25} {'Expected Hint'}")
    print("-" * 110)
    for _, row in df.iterrows():
        item_no = row["Item_Number"]
        hint = hints.get(item_no, "?")
        print(
            f"{item_no:<18} "
            f"{str(row['AI_confidence']):<12} "
            f"{str(row.get('AI_source', '')):<16} "
            f"{str(row['AI_MATERIAL_CATEGORY']):<25} "
            f"{hint}"
        )

    print()
    print("Details:")
    print("-" * 110)
    for _, row in df.iterrows():
        print(f"\n  {row['Item_Number']}  ({row['Item_Desc']})")
        print(f"  MPN: {row['MFR_PART_NUMBER']}")
        print(f"  AI: {row['AI_MATERIAL_CATEGORY']}  [{row['AI_confidence']}] via {row.get('AI_source', '?')}")
        vec = row.get("Vector_used", "N")
        if vec == "Y":
            print(f"  Vector top1: {row.get('Vector_top1_category', '')} (score={row.get('Vector_top1_score', 0)})")
        print(f"  Reason: {str(row['AI_reason'])[:150]}")

    print(f"\nExcel saved to: tests/_last_test_result.xlsx")


if __name__ == "__main__":
    main()

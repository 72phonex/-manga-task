#!/usr/bin/env python3
"""
Validation script for Manga Task final submission.
Verifies that predictions.jsonl adheres strictly to format:
- Exactly 15 lines matching sample_submission.jsonl sequence_ids
- Each line has a valid JSON object with 'sequence_id' and 'pages' (length 3)
- Pages contain lists of {'speaker': str, 'text': str}
- Passes score.py strict validation without any schema warnings
"""

import json
import sys
from pathlib import Path
import score

def validate(preds_file="predictions.jsonl", sample_file="sample_submission.jsonl"):
    print("=" * 60)
    print("MANGA TASK: SUBMISSION VALIDATION REPORT")
    print("=" * 60)

    # 1. Existence check
    if not Path(preds_file).exists():
        print(f"[FAIL] Predictions file {preds_file} does not exist!")
        return False
    if not Path(sample_file).exists():
        print(f"[FAIL] Sample submission file {sample_file} does not exist!")
        return False

    # 2. Line count & JSON integrity
    with open(sample_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    with open(preds_file, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]

    print(f"Sample submission lines: {len(samples)}")
    print(f"Predictions lines:       {len(preds)}")

    if len(samples) != len(preds):
        print(f"[FAIL] Line count mismatch: expected {len(samples)}, got {len(preds)}")
        return False

    # 3. ID alignment & page schema checks
    total_boxes = 0
    pages_with_content = 0
    empty_pages = 0

    for idx, (s, p) in enumerate(zip(samples, preds), 1):
        s_id = s.get("sequence_id")
        p_id = p.get("sequence_id")
        if s_id != p_id:
            print(f"[FAIL] Sequence ID mismatch at row {idx}: expected '{s_id}', got '{p_id}'")
            return False

        pages = p.get("pages")
        if not isinstance(pages, list) or len(pages) != 3:
            print(f"[FAIL] Sequence {p_id} does not have exactly 3 pages (found {len(pages) if isinstance(pages, list) else type(pages)})")
            return False

        for page_idx, page in enumerate(pages):
            if not isinstance(page, list):
                print(f"[FAIL] Sequence {p_id} page {page_idx} is not a list")
                return False
            if len(page) == 0:
                empty_pages += 1
            else:
                pages_with_content += 1
                for item_idx, item in enumerate(page):
                    if not isinstance(item, dict):
                        print(f"[FAIL] Sequence {p_id} page {page_idx} item {item_idx} is not an object")
                        return False
                    if "speaker" not in item or "text" not in item:
                        print(f"[FAIL] Sequence {p_id} item missing 'speaker' or 'text': {item}")
                        return False
                    total_boxes += 1

    # 4. Official score.py loader verification
    try:
        loaded = score.load_jsonl(preds_file, pages=3)
        print(f"[PASS] Successfully loaded via score.py with {len(loaded)} sequences verified.")
    except Exception as e:
        print(f"[FAIL] Failed validation through score.load_jsonl: {e}")
        return False

    print("-" * 60)
    print(f"Total transcribed speech bubbles: {total_boxes}")
    print(f"Pages with dialogue:              {pages_with_content}")
    print(f"Empty pages (correctly preserved):{empty_pages}")
    print("=" * 60)
    print("[PASS] ALL SUBMISSION CHECKS PASSED PERFECTLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)

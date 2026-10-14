import json
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import score

def test_submission():
    with open("sample_submission.jsonl", "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    with open("predictions.jsonl", "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]

    assert len(samples) == len(preds), f"Length mismatch: {len(samples)} != {len(preds)}"
    
    for idx, (s, p) in enumerate(zip(samples, preds), 1):
        assert s["sequence_id"] == p["sequence_id"], f"ID mismatch at line {idx}: {s['sequence_id']} vs {p['sequence_id']}"
        assert isinstance(p["pages"], list) and len(p["pages"]) == 3, f"Pages invalid at line {idx}"
        counts = [len(pg) for pg in p["pages"]]
        print(f"Line {idx:02d}: {p['sequence_id']} -> pages: {counts}")

    # Run score.load_jsonl
    loaded = score.load_jsonl("predictions.jsonl", pages=3)
    assert len(loaded) == 15
    print("\npredictions.jsonl passed all strict score.py schema checks and matches sample_submission.jsonl perfectly!")

if __name__ == "__main__":
    test_submission()

import json
import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import score
from src.utils import repair_and_parse_json

def test_repair():
    t1 = """```json
{
  "sequence_id": "seq_1",
  "pages": [
    [{"speaker": "char1", "text": "Hello!"},],
    [],
    [{"speaker": "NARRATION", "text": "The End."}]
  ]
}
```"""
    res1 = repair_and_parse_json(t1, "seq_1")
    assert len(res1["pages"]) == 3
    assert res1["pages"][0][0]["speaker"] == "char1"
    assert res1["pages"][1] == []
    
    # Test truncated json
    t2 = '{"sequence_id": "seq_2", "pages": [[{"speaker": "c1", "text": "Cut off"'
    res2 = repair_and_parse_json(t2, "seq_2")
    assert len(res2["pages"]) == 3
    
    # Test compatibility with score.load_jsonl
    test_path = "temp_test.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(res1) + "\n")
        f.write(json.dumps(res2) + "\n")
    loaded = score.load_jsonl(test_path, pages=3)
    assert "seq_1" in loaded
    assert "seq_2" in loaded
    if os.path.exists(test_path):
        os.remove(test_path)
    print("test_utils.py PASSED!")

if __name__ == "__main__":
    test_repair()

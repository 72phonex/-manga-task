import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decode_guards import has_repeat_loop, dedupe_page, problems, pick_best

def b(t, s="character1"): return {"speaker": s, "text": t}

def test_repeat_loop_detected():
    raw = '{"pages":[[' + ",".join('{"speaker":"character1","text":"HELLO THERE"}' for _ in range(3))
    assert has_repeat_loop(raw)

def test_no_false_positive_on_distinct_or_short():
    raw = '[{"text":"A"},{"text":"B"},{"text":"C"}]'
    assert not has_repeat_loop(raw)
    assert not has_repeat_loop('[{"text":"..."},{"text":"..."},{"text":"..."}]')  # punctuation bubbles are legit

def test_hard_cap():
    assert has_repeat_loop(",".join(f'{{"text":"line {i}"}}' for i in range(61)))

def test_dedupe_keeps_first_and_short_texts():
    page = [b("HELLO"), b("hello"), b("?!"), b("?!"), b("BYE")]
    assert [x["text"] for x in dedupe_page(page)] == ["HELLO", "?!", "?!", "BYE"]

def test_problems_and_pick_best():
    ok = {"sequence_id": "x", "pages": [[b("hi")], [], []]}
    empty = {"sequence_id": "x", "pages": [[], [], []]}
    assert problems(ok) == [] and "all_empty" in problems(empty)
    many = {"sequence_id": "x", "pages": [[b("a", f"c{i}") for i in range(13)], [], []]}
    assert "too_many_speakers" in problems(many)
    assert pick_best([empty, many, ok]) is ok

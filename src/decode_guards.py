"""Torch-free decoding guards for the manga pipeline (testable without a GPU)."""
import re

MAX_PER_PAGE = 19          # max bubbles on any development page
MAX_SPEAKERS = 12          # more distinct labels than this on one sequence is implausible
MIN_DEDUPE_LEN = 4

_TEXT_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')


def has_repeat_loop(raw: str, repeats: int = 3, hard_cap: int = 60) -> bool:
    """True when generation looks stuck: the last `repeats` bubble texts are identical,
    or the output already holds more bubbles than any 3-page sequence could."""
    texts = [t.strip().lower() for t in _TEXT_RE.findall(raw)]
    if len(texts) > hard_cap:
        return True
    if len(texts) >= repeats:
        tail = texts[-repeats:]
        if len(tail[0]) >= MIN_DEDUPE_LEN and all(t == tail[0] for t in tail):
            return True
    return False


def dedupe_page(page: list) -> list:
    seen, keep = set(), []
    for b in page:
        k = b["text"].strip().lower()
        if len(k) >= MIN_DEDUPE_LEN and k in seen:
            continue
        seen.add(k)
        keep.append(b)
    return keep[:MAX_PER_PAGE]


def dedupe_sequence(seq: dict) -> dict:
    return {"sequence_id": seq["sequence_id"], "pages": [dedupe_page(p) for p in seq["pages"]]}


def problems(seq: dict) -> list:
    """Plausibility problems of a (deduped) parsed sequence; empty list means it looks fine."""
    out = []
    pages = seq["pages"]
    if not any(pages):
        out.append("all_empty")
    for i, p in enumerate(pages):
        if len(p) >= MAX_PER_PAGE:
            out.append(f"page{i + 1}_at_cap")
    speakers = {b["speaker"] for p in pages for b in p}
    if len(speakers) > MAX_SPEAKERS:
        out.append("too_many_speakers")
    return out


def pick_best(candidates: list) -> dict:
    """candidates: list of parsed+deduped sequences. Fewest problems wins, then most bubbles."""
    return min(candidates, key=lambda s: (len(problems(s)), -sum(len(p) for p in s["pages"])))

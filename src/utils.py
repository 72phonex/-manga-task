import re
import json
import unicodedata

def clean_text_for_json(text: str) -> str:
    """Strip control characters and excessive whitespaces."""
    if not isinstance(text, str):
        return ""
    # Remove control characters except normal space
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch.isspace())
    return text.strip()

def repair_and_parse_json(raw_str: str, sequence_id: str, num_pages: int = 3) -> dict:
    """
    Robustly extract, repair, and validate sequence JSON conforming to score.py requirements:
    {"sequence_id": sequence_id, "pages": [[{"speaker": "...", "text": "..."}, ...], [], []]}
    """
    fallback = {"sequence_id": sequence_id, "pages": [[] for _ in range(num_pages)]}
    if not raw_str or not isinstance(raw_str, str):
        return fallback

    # 1. Strip markdown fences
    text = raw_str.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Look for first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    # 2. Attempt direct json.loads
    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        # Basic regex repair for common LLM JSON syntax issues
        # Remove trailing commas before } or ]
        repaired = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            parsed = json.loads(repaired)
        except Exception:
            # Fallback attempt: balance braces if output was truncated
            repaired_balanced = repaired
            open_braces = repaired_balanced.count("{") - repaired_balanced.count("}")
            open_brackets = repaired_balanced.count("[") - repaired_balanced.count("]")
            if open_brackets > 0:
                repaired_balanced += "]" * open_brackets
            if open_braces > 0:
                repaired_balanced += "}" * open_braces
            try:
                parsed = json.loads(repaired_balanced)
            except Exception:
                pass

    if not isinstance(parsed, dict):
        return fallback

    # 3. Enforce schema
    pages_raw = parsed.get("pages")
    if not isinstance(pages_raw, list):
        return fallback

    clean_pages = []
    for page_idx in range(num_pages):
        if page_idx < len(pages_raw) and isinstance(pages_raw[page_idx], list):
            clean_page = []
            for item in pages_raw[page_idx]:
                if not isinstance(item, dict):
                    continue
                speaker = item.get("speaker")
                item_text = item.get("text")
                if not isinstance(speaker, str) or not speaker.strip():
                    speaker = "character1"
                else:
                    speaker = clean_text_for_json(speaker.strip())
                    if not speaker:
                        speaker = "character1"

                if not isinstance(item_text, str) or not item_text.strip():
                    continue
                item_text = clean_text_for_json(item_text.strip())
                if not item_text:
                    continue

                clean_page.append({"speaker": speaker, "text": item_text})
            clean_pages.append(clean_page)
        else:
            clean_pages.append([])

    return {
        "sequence_id": sequence_id,
        "pages": clean_pages
    }

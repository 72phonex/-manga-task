import re
import unicodedata
from collections import OrderedDict

PUNCT_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "…": "...", "–": "-", "—": "-", "−": "-"})

KNOWN_SFX_WORDS = {
    "thud", "crash", "gulp", "swoosh", "dooon", "bam", "tap", "rattle",
    "rumble", "click", "clang", "creak", "phew", "pant", "gasp", "wham",
    "boom", "kaboom", "clack", "rustle", "sizzle", "splash", "fwip", "shhk",
    "doki", "ba-dump", "badump", "thump", "bonk", "whoosh", "smack", "shing",
    "clatter", "step", "footsteps", "snicker", "giggle", "sigh", "chuckle",
}

META_PATTERNS = [
    re.compile(r"^(?:page\s*\d+|chapter\s*\d+|vol(?:ume)?\s*\d+)", re.IGNORECASE),
    re.compile(r"^(?:to be continued|the end|fin|end of chapter)", re.IGNORECASE),
    re.compile(r"^(?:translated by|scans?|scanlation|edited by|typeset)", re.IGNORECASE),
]

def clean_text(text: str) -> str:
    """Normalize text and remove control characters."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).translate(PUNCT_TRANS)
    text = "".join(c for c in text if unicodedata.category(c) != "Cc" or c.isspace())
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_sfx_or_meta(text: str) -> bool:
    """Detect if an item is likely a sound effect, reaction noise, or scanlator/meta text."""
    clean = clean_text(text)
    if not clean:
        return True

    # 1. Check meta patterns
    for pat in META_PATTERNS:
        if pat.search(clean):
            return True

    # 2. Check standalone sound effect wrapped in asterisks e.g. *thud* or *swoosh*
    if re.fullmatch(r"\*[a-zA-Z\s'-]+\*", clean):
        return True

    # 3. Check single isolated SFX tokens
    token = clean.strip("!?.~* \t\n\r").lower()
    if token in KNOWN_SFX_WORDS:
        return True

    # 4. Check sound effect sound representations (e.g. "BZZZZT", "KABOOOM", "ZZZ")
    if len(token) >= 3 and len(set(token)) <= 2 and token.isalpha():
        return True

    return False

def normalize_speaker(speaker: str) -> str:
    """Normalize speaker token, preserving NARRATION."""
    if not isinstance(speaker, str) or not speaker.strip():
        return "character1"
    speaker = clean_text(speaker.strip())
    speaker_upper = speaker.upper()
    if "NARRAT" in speaker_upper or "EXPOS" in speaker_upper:
        return "NARRATION"
    return speaker

def post_process_sequence(seq_dict: dict) -> dict:
    """
    Apply post-processing harness to a single sequence dict:
    1. Enforce 3 pages schema
    2. Drop SFX-looking, meta, or empty items
    3. Clean and merge wrapped text
    4. Canonicalize speaker cluster IDs across the sequence (character1, character2, ... and NARRATION)
    """
    seq_id = seq_dict.get("sequence_id", "unknown_seq")
    pages_raw = seq_dict.get("pages", [[], [], []])
    if not isinstance(pages_raw, list):
        pages_raw = [[], [], []]

    # Speaker mapping across the 3 pages of the sequence
    speaker_map = OrderedDict()
    char_counter = 1

    processed_pages = []
    for page_idx in range(3):
        if page_idx >= len(pages_raw) or not isinstance(pages_raw[page_idx], list):
            processed_pages.append([])
            continue

        raw_items = pages_raw[page_idx]
        filtered_items = []

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            text = clean_text(item.get("text", ""))
            speaker = normalize_speaker(item.get("speaker", ""))

            # Filter out empty or SFX
            if not text:
                continue
            if is_sfx_or_meta(text):
                continue

            # Map speaker label consistently
            if speaker == "NARRATION":
                mapped_speaker = "NARRATION"
            else:
                if speaker not in speaker_map:
                    speaker_map[speaker] = f"character{char_counter}"
                    char_counter += 1
                mapped_speaker = speaker_map[speaker]

            # Merge wrapped line if same speaker and previous line ended without terminal punctuation
            if (
                filtered_items
                and filtered_items[-1]["speaker"] == mapped_speaker
                and not filtered_items[-1]["text"].endswith((".", "!", "?", '."', '!"', '?"', "...", "—"))
                and (text[0].islower() or filtered_items[-1]["text"].endswith(("-", ",")))
            ):
                filtered_items[-1]["text"] = clean_text(f"{filtered_items[-1]['text']} {text}")
            else:
                filtered_items.append({"speaker": mapped_speaker, "text": text})

        processed_pages.append(filtered_items)

    return {
        "sequence_id": seq_id,
        "pages": processed_pages,
    }

def post_process_file(input_file: str, output_file: str):
    """Process an entire JSONL predictions file through the harness."""
    import json
    from pathlib import Path

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            processed = post_process_sequence(data)
            f_out.write(json.dumps(processed, ensure_ascii=False) + "\n")
    print(f"Post-processing complete! Written to {output_file}")

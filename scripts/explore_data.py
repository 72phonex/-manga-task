import json
from collections import Counter

with open("development/labels.jsonl", "r", encoding="utf-8") as f:
    dev = [json.loads(line) for line in f if line.strip()]

print(f"Total dev sequences: {len(dev)}")

# 1. Inspect speaker distribution
speaker_counts = Counter()
page_lengths = []
bubble_lengths = []
empty_page_seqs = []

for s in dev:
    seq_id = s["sequence_id"]
    for pi, page in enumerate(s["pages"]):
        page_lengths.append(len(page))
        if len(page) == 0:
            empty_page_seqs.append((seq_id, pi + 1))
        for item in page:
            speaker_counts[item["speaker"]] += 1
            bubble_lengths.append(len(item["text"].split()))

print("Speaker counts:", speaker_counts)
print(f"Empty pages: {len(empty_page_seqs)}: {empty_page_seqs}")
print(f"Page length stats: min={min(page_lengths)}, max={max(page_lengths)}, avg={sum(page_lengths)/len(page_lengths):.2f}")
print(f"Bubble word count stats: min={min(bubble_lengths)}, max={max(bubble_lengths)}, avg={sum(bubble_lengths)/len(bubble_lengths):.2f}")

# 2. Check narration examples
print("\n--- Narration Examples ---")
narr_samples = []
for s in dev:
    for pi, page in enumerate(s["pages"]):
        for item in page:
            if item["speaker"] == "NARRATION":
                narr_samples.append((s["sequence_id"], pi + 1, item["text"]))
                if len(narr_samples) >= 5:
                    break
        if len(narr_samples) >= 5:
            break
    if len(narr_samples) >= 5:
        break
for seq_id, pnum, text in narr_samples:
    print(f"[{seq_id} Page {pnum}]: {text}")

# 3. Check punctuation-only / short bubbles
print("\n--- Short / Punctuation-only Bubbles ---")
short_samples = []
for s in dev:
    for pi, page in enumerate(s["pages"]):
        for item in page:
            if len(item["text"]) <= 3:
                short_samples.append((item["speaker"], item["text"]))
                if len(short_samples) >= 10:
                    break
print("Short samples:", short_samples)

# 4. Check test sequences in sequences.json
with open("sequences.json", "r", encoding="utf-8") as f:
    seqs = json.load(f)

dev_seqs = [s for s in seqs if s["split"] == "development"]
test_seqs = [s for s in seqs if s["split"] == "test"]
print(f"\nSequences.json: {len(dev_seqs)} development, {len(test_seqs)} test")

with open("sample_submission.jsonl", "r", encoding="utf-8") as f:
    sub = [json.loads(line) for line in f if line.strip()]
print(f"Sample submission count: {len(sub)}")
test_ids = [s["sequence_id"] for s in test_seqs]
sub_ids = [s["sequence_id"] for s in sub]
assert test_ids == sub_ids, "Test IDs mismatch between sequences.json and sample_submission.jsonl!"
print("Test IDs perfectly match sample_submission.jsonl!")

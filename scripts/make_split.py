import json
import random
from pathlib import Path

def make_split(seed=42, val_count=16):
    labels_path = Path("development/labels.jsonl")
    with open(labels_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    seq_ids = sorted([r["sequence_id"] for r in records])
    rng = random.Random(seed)
    val_ids = sorted(rng.sample(seq_ids, val_count))
    train_ids = sorted([sid for sid in seq_ids if sid not in val_ids])
    
    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)
    
    with open(splits_dir / "train_ids.json", "w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)
    with open(splits_dir / "val_ids.json", "w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)
        
    # Also write val_labels.jsonl and train_labels.jsonl for fast scoring with score.py
    val_records = [r for r in records if r["sequence_id"] in set(val_ids)]
    train_records = [r for r in records if r["sequence_id"] in set(train_ids)]
    
    with open(splits_dir / "val_labels.jsonl", "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(splits_dir / "train_labels.jsonl", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Created fixed train/val split with seed={seed}:")
    print(f"  Train: {len(train_ids)} sequences ({splits_dir / 'train_labels.jsonl'})")
    print(f"  Val:   {len(val_ids)} sequences ({splits_dir / 'val_labels.jsonl'})")
    return train_ids, val_ids

if __name__ == "__main__":
    make_split()

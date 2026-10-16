#!/usr/bin/env python3
"""Inference v2 for The Manga Task: same 3-page prompt/adapter as v1, plus
  * a stopping criterion that kills repetition loops mid-generation,
  * per-sequence plausibility checks with up to 2 sampled retries,
  * duplicate-bubble removal and the fixed SFX harness.

Usage (validation):  python -m src.infer_v2 --split val  --output results/lora_v2_val.jsonl
Usage (test):        python -m src.infer_v2 --split test --output predictions.jsonl
Try --max-pixels 501760 (what the adapter was trained at) vs 1003520 on validation and keep the better one.
"""
import argparse, json, os, sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.decode_guards import has_repeat_loop, dedupe_sequence, problems, pick_best
from src.utils import repair_and_parse_json

ATTEMPTS = [  # (do_sample, temperature, top_p, repetition_penalty)
    (False, None, None, 1.05),
    (True, 0.3, 0.9, 1.10),
    (True, 0.5, 0.9, 1.10),
]


def make_stopper(tokenizer, prompt_len, check_every=24):
    import torch
    from transformers import StoppingCriteria

    class RepeatStopper(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            n = input_ids.shape[1] - prompt_len
            stop = n > 0 and n % check_every == 0 and has_repeat_loop(
                tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True))
            return torch.tensor([stop], device=input_ids.device, dtype=torch.bool)

    return RepeatStopper()


def generate_once(model, processor, inputs, attempt, max_new_tokens):
    import torch
    from transformers import StoppingCriteriaList
    do_sample, temp, top_p, rep = ATTEMPTS[attempt]
    prompt_len = inputs.input_ids.shape[1]
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, repetition_penalty=rep,
                  stopping_criteria=StoppingCriteriaList([make_stopper(processor.tokenizer, prompt_len)]))
    if do_sample:
        kwargs.update(temperature=temp, top_p=top_p)
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    return processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]


def transcribe(model, processor, item, max_new_tokens=1536, max_retries=2):
    from qwen_vl_utils import process_vision_info
    from src.infer import build_prompt_for_sequence
    seq_id = item["sequence_id"]
    messages = build_prompt_for_sequence(seq_id, item["images"])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    candidates = []
    for attempt in range(max_retries + 1):
        raw = generate_once(model, processor, inputs, attempt, max_new_tokens)
        cand = dedupe_sequence(repair_and_parse_json(raw, seq_id, num_pages=3))
        candidates.append(cand)
        if not problems(cand):
            break
        print(f"  [{seq_id}] attempt {attempt}: {problems(cand)} -> retrying" if attempt < max_retries else
              f"  [{seq_id}] attempt {attempt}: {problems(cand)} -> keeping best")
    return pick_best(candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter-path", default="outputs/adapter")
    ap.add_argument("--sequences", default="sequences.json")
    ap.add_argument("--val-ids", default="splits/val_ids.json")
    ap.add_argument("--sample-submission", default="sample_submission.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-pixels", type=int, default=501760)
    ap.add_argument("--min-pixels", type=int, default=200704)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--no-harness", action="store_true")
    args = ap.parse_args()

    from tqdm import tqdm
    from src.infer import load_vlm_model
    from src.harness import post_process_sequence

    by_id = {s["sequence_id"]: s for s in json.load(open(args.sequences, encoding="utf-8"))}
    if args.split == "test":
        ids = [json.loads(l)["sequence_id"] for l in open(args.sample_submission) if l.strip()]
    else:
        ids = json.load(open(args.val_ids))
    items = [by_id[i] for i in ids]
    print(f"{len(items)} sequences | max_pixels={args.max_pixels}")

    model, processor = load_vlm_model(args.model_path, args.adapter_path, True,
                                      min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in tqdm(items):
            pred = transcribe(model, processor, item, args.max_new_tokens)
            if not args.no_harness:
                pred = post_process_sequence(pred)
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            f.flush()

    if args.split == "test":
        import score
        loaded = score.load_jsonl(args.output, pages=3)
        assert list(loaded.keys()) == ids, "order/IDs differ from sample_submission.jsonl"
        print(f"OK: {args.output} has {len(loaded)} sequences in the required order.")
    else:
        print(f"Score it: python score.py --references splits/val_labels.jsonl --predictions {args.output} --output results/v2.json")


if __name__ == "__main__":
    main()

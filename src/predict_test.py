#!/usr/bin/env python3
"""Run best model (LoRA + Harness) on the 15 test sequences and generate predictions.jsonl."""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import score
from src.infer import load_vlm_model, build_prompt_for_sequence
from src.utils import repair_and_parse_json
from src.harness import post_process_sequence
from qwen_vl_utils import process_vision_info

def predict_test(
    model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    adapter_path: str = "outputs/adapter",
    sequences_path: str = "sequences.json",
    sample_sub_path: str = "sample_submission.jsonl",
    output_path: str = "predictions.jsonl",
    max_pixels: int = 501760,
    min_pixels: int = 200704,
    max_new_tokens: int = 1536,
):
    print("=" * 60)
    print("TEST SET PREDICTION PIPELINE (Best Configuration: LoRA + Harness)")
    print("=" * 60)

    # 1. Load sample submission to get exact sequence_id ordering
    print(f"Reading sequence order from {sample_sub_path}...")
    with open(sample_sub_path, "r", encoding="utf-8") as f:
        ordered_test_ids = [json.loads(line)["sequence_id"] for line in f if line.strip()]
    print(f"Found {len(ordered_test_ids)} test sequences in sample submission.")

    # 2. Load sequence metadata
    with open(sequences_path, "r", encoding="utf-8") as f:
        all_sequences = {s["sequence_id"]: s for s in json.load(f)}

    test_sequences = []
    for sid in ordered_test_ids:
        if sid in all_sequences:
            test_sequences.append(all_sequences[sid])
        else:
            raise ValueError(f"Test sequence {sid} not found in {sequences_path}!")

    # 3. Load model with adapter
    print(f"Loading model {model_path} with adapter {adapter_path}...")
    model, processor = load_vlm_model(
        model_path=model_path,
        adapter_path=adapter_path,
        load_in_4bit=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # 4. Generate predictions
    predictions = []
    print("\nGenerating test predictions...")
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in tqdm(test_sequences, desc="Predicting test sequences"):
            seq_id = item["sequence_id"]
            image_paths = item["images"]

            messages = build_prompt_for_sequence(seq_id, image_paths)
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.05,
                )

            prompt_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, prompt_len:]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Parse and repair JSON
            raw_parsed = repair_and_parse_json(output_text, seq_id, num_pages=3)

            # Apply post-processing harness
            clean_parsed = post_process_sequence(raw_parsed)

            f_out.write(json.dumps(clean_parsed, ensure_ascii=False) + "\n")
            f_out.flush()
            predictions.append(clean_parsed)

    # 5. Strict schema validation using score.py's load_jsonl
    print(f"\nValidating {output_path} with score.load_jsonl...")
    loaded = score.load_jsonl(output_path, pages=3)
    assert len(loaded) == len(ordered_test_ids), f"Expected {len(ordered_test_ids)} entries, got {len(loaded)}"
    assert list(loaded.keys()) == ordered_test_ids, "Sequence order or IDs do not match sample_submission.jsonl!"

    print("=" * 60)
    print(f"SUCCESS: {output_path} generated and validated! Exactly {len(loaded)} sequences.")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Generate test predictions for The Manga Task.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="outputs/adapter")
    parser.add_argument("--sequences", type=str, default="sequences.json")
    parser.add_argument("--sample-submission", type=str, default="sample_submission.jsonl")
    parser.add_argument("--output", type=str, default="predictions.jsonl")
    args = parser.parse_args()

    predict_test(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        sequences_path=args.sequences,
        sample_sub_path=args.sample_submission,
        output_path=args.output,
    )

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inference script for The Manga Task using Qwen2.5-VL."""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from peft import PeftModel

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils import repair_and_parse_json

SYSTEM_PROMPT = """You are an expert manga transcription assistant.
You are given 3 consecutive English manga pages (Page 1, Page 2, Page 3).
Transcribe the speech and narration across all 3 pages into strict JSON.

Output format:
{
  "sequence_id": "<sequence_id>",
  "pages": [
    [
      {"speaker": "character1", "text": "DIALOGUE"},
      {"speaker": "character2", "text": "REPLY"}
    ],
    [],
    [
      {"speaker": "NARRATION", "text": "NARRATION TEXT"}
    ]
  ]
}

CRITICAL RULES:
1. READING ORDER: Transcribe each page in traditional manga reading order (right-to-left, top-to-bottom across panels and speech bubbles).
2. SPEAKER CONSISTENCY: Keep speaker identifiers consistent across all 3 pages (e.g. character1, character2, character3). For narrator or exposition boxes, use EXACTLY "NARRATION".
3. INCLUDE:
   - Dialogue spoken by characters in speech bubbles
   - Thoughts and internal monologues (attributed to the thinker)
   - Narration boxes (speaker: "NARRATION")
   - Clear spoken screams, grunts, or vocalizations (e.g. "WOAH!!", "AHH", "HUH?")
   - Punctuation-only bubbles (e.g. "...", "?!", "!?")
4. EXCLUDE:
   - Sound effects (SFX) / onomatopoeia and sound translations
   - Tiny background reaction / breath text outside speech bubbles
   - Chapter titles, series logos, credits, page numbers, watermarks, scanlator notes
   - Environmental text on signs, clothes, posters, background objects
   - Text inside letters, books, phones, or documents
5. EMPTY PAGES: If a page has no dialogue, thoughts, or narration, use an empty list [] for that page. Do NOT invent text on empty pages.

Respond with ONLY the valid JSON object."""

def load_vlm_model(
    model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    adapter_path: str = None,
    load_in_4bit: bool = True,
    device: str = "cuda",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 640 * 28 * 28,
):
    print(f"Loading processor for {model_path} (min_pixels={min_pixels}, max_pixels={max_pixels})...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )

    bnb_config = None
    if load_in_4bit and device == "cuda":
        print("Using 4-bit quantization (BitsAndBytes)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"Loading base model {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading PEFT adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    else:
        model.eval()

    return model, processor

def build_prompt_for_sequence(sequence_id: str, image_paths: list) -> list:
    content = []
    for img_path in image_paths:
        abs_img_path = str(Path(img_path).resolve())
        content.append({"type": "image", "image": abs_img_path})
    
    user_prompt = f"{SYSTEM_PROMPT}\n\nTranscribe the sequence with sequence_id: \"{sequence_id}\"."
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "user", "content": content}
    ]
    return messages

def run_inference(
    model,
    processor,
    sequences: list,
    output_path: str,
    max_new_tokens: int = 1536,
    post_process_fn = None,
):
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"Starting inference on {len(sequences)} sequences...")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in tqdm(sequences, desc="Transcribing manga sequences"):
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
            )
            inputs = inputs.to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.05,
                )

            # Strip input prompt tokens from output
            prompt_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, prompt_len:]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Parse and repair JSON
            parsed = repair_and_parse_json(output_text, seq_id, num_pages=3)

            # Optional post-processing harness
            if post_process_fn:
                parsed = post_process_fn(parsed)

            f_out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            f_out.flush()
            results.append(parsed)

    print(f"Inference complete! Results written to {output_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run manga sequence transcription inference.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--sequences", type=str, default="sequences.json")
    parser.add_argument("--split-ids", type=str, default="splits/val_ids.json")
    parser.add_argument("--output", type=str, default="results/baseline_preds.jsonl")
    parser.add_argument("--max-pixels", type=int, default=501760)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--load-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_4bit", action="store_false")
    parser.add_argument("--apply-harness", action="store_true", default=False)
    args = parser.parse_args()

    # Load sequences
    with open(args.sequences, "r", encoding="utf-8") as f:
        all_sequences = json.load(f)

    if args.split_ids and os.path.exists(args.split_ids):
        with open(args.split_ids, "r", encoding="utf-8") as f:
            selected_ids = set(json.load(f))
        target_sequences = [s for s in all_sequences if s["sequence_id"] in selected_ids]
    else:
        target_sequences = all_sequences

    print(f"Target sequence count: {len(target_sequences)}")

    post_fn = None
    if args.apply_harness:
        from src.harness import post_process_sequence
        post_fn = post_process_sequence
        print("Enabled post-processing harness.")

    model, processor = load_vlm_model(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_4bit,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    run_inference(
        model=model,
        processor=processor,
        sequences=target_sequences,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        post_process_fn=post_fn,
    )

if __name__ == "__main__":
    main()

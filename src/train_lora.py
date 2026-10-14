#!/usr/bin/env python3
"""Fine-tune Qwen2.5-VL-3B-Instruct on manga sequence transcription using LoRA/PEFT."""
import argparse
import json
import math
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.infer import SYSTEM_PROMPT

def train_lora(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    train_labels_path: str = "splits/train_labels.jsonl",
    sequences_path: str = "sequences.json",
    output_dir: str = "outputs/adapter",
    results_dir: str = "results",
    epochs: int = 2,
    lr: float = 1e-4,
    grad_accum_steps: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    min_pixels: int = 64 * 28 * 28,
    max_pixels: int = 128 * 28 * 28,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading sequences from {sequences_path}...")
    with open(sequences_path, "r", encoding="utf-8") as f:
        seqs_meta = {s["sequence_id"]: s for s in json.load(f)}

    print(f"Loading train records from {train_labels_path}...")
    with open(train_labels_path, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]

    print(f"Total training sequences: {len(train_records)}")

    # 1. Setup Processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )

    # 2. Setup 4-bit Quantized Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading base model {model_name} in 4-bit...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Freeze vision encoder to conserve VRAM
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        for param in model.model.visual.parameters():
            param.requires_grad = False

    model.config.use_cache = False
    model.enable_input_require_grads()
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        model.model.language_model.gradient_checkpointing_enable()

    # 3. Setup LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Setup Optimizer and Scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    total_steps = (len(train_records) * epochs) // grad_accum_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)

    # 5. Training Loop
    training_logs = []
    global_step = 0
    model.train()

    print(f"\nBeginning LoRA fine-tuning for {epochs} epochs ({total_steps} optimizer steps)...")
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        optimizer.zero_grad()

        # Shuffle training records deterministically per epoch
        perm = torch.randperm(len(train_records), generator=torch.Generator().manual_seed(42 + epoch))
        shuffled_records = [train_records[i] for i in perm.tolist()]

        pbar = tqdm(shuffled_records, desc=f"Epoch {epoch}/{epochs}")
        for step_idx, record in enumerate(pbar):
            seq_id = record["sequence_id"]
            seq_info = seqs_meta[seq_id]
            image_paths = [str(Path(p).resolve()) for p in seq_info["images"]]

            user_prompt = f"{SYSTEM_PROMPT}\n\nTranscribe the sequence with sequence_id: \"{seq_id}\"."
            target_json = json.dumps(record, ensure_ascii=False)

            content = [{"type": "image", "image": img} for img in image_paths]
            content.append({"type": "text", "text": user_prompt})

            # Format chat
            messages_prompt = [{"role": "user", "content": content}]
            prompt_text = processor.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True
            )

            messages_full = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": target_json}]},
            ]
            full_text = processor.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False
            )

            image_inputs, video_inputs = process_vision_info(messages_full)
            inputs_full = processor(
                text=[full_text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
            ).to(model.device)

            inputs_prompt = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
            ).to(model.device)

            prompt_len = inputs_prompt.input_ids.shape[1]
            labels = inputs_full.input_ids.clone()
            labels[:, :prompt_len] = -100 # Mask prompt tokens from loss

            torch.cuda.empty_cache()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs_full, labels=labels)
                raw_loss = outputs.loss
                loss = raw_loss / grad_accum_steps

            loss.backward()
            epoch_losses.append(raw_loss.item())
            pbar.set_postfix({"loss": f"{raw_loss.item():.4f}", "avg_loss": f"{sum(epoch_losses)/len(epoch_losses):.4f}"})

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(shuffled_records):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                training_logs.append({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": raw_loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                })

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch} finished! Average loss: {avg_epoch_loss:.4f}")

    # 6. Save Adapter
    print(f"\nSaving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Save log
    train_log_file = Path(results_dir) / "train_log.json"
    with open(train_log_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "epochs": epochs,
            "train_samples": len(train_records),
            "final_epoch_loss": avg_epoch_loss,
            "history": training_logs,
        }, f, indent=2)

    print(f"Training log written to {train_log_file}")
    print("Fine-tuning complete!")

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter on Manga dataset.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train-labels", type=str, default="splits/train_labels.jsonl")
    parser.add_argument("--sequences", type=str, default="sequences.json")
    parser.add_argument("--output-dir", type=str, default="outputs/adapter")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    train_lora(
        model_name=args.model_name,
        train_labels_path=args.train_labels,
        sequences_path=args.sequences,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        lr=args.lr,
        grad_accum_steps=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

if __name__ == "__main__":
    main()

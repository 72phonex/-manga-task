#!/usr/bin/env python3
"""Environment and GPU validation script for The Manga Task pipeline."""
import sys
import os
import json
from pathlib import Path

def check_env():
    print("=" * 60)
    print("ENVIRONMENT & HARDWARE CHECK")
    print("=" * 60)
    
    # 1. Python
    print(f"[1] Python: {sys.version.split()[0]} ({sys.executable})")
    
    # Ensure repo root is on sys.path
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    # 2. PyTorch & CUDA
    import torch
    print(f"[2] PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"    CUDA available: {cuda_available}")
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"    Device count: {device_count}")
        print(f"    Device 0: {device_name}")
        print(f"    Compute capability: {capability}")
        print(f"    Total VRAM: {vram_gb:.2f} GB")
        
        # Test tensor memory allocation
        x = torch.randn(100, 100, device="cuda")
        print(f"    CUDA tensor allocation test: SUCCESS ({x.sum().item():.2f})")
    else:
        print("    WARNING: CUDA is NOT available! Check PyTorch build.")
    
    # 3. BitsAndBytes 4-bit test
    print("\n[3] BitsAndBytes Check:")
    try:
        import bitsandbytes as bnb
        print(f"    bitsandbytes version: {bnb.__version__}")
        if cuda_available:
            l4 = bnb.nn.Linear4bit(64, 64).cuda()
            dummy = torch.randn(2, 64, device="cuda")
            out = l4(dummy)
            print(f"    4-bit quantization kernel test: SUCCESS (output shape {out.shape})")
        else:
            print("    Skipping 4-bit CUDA kernel check (no CUDA).")
    except Exception as e:
        print(f"    BitsAndBytes test failed or not available: {e}")
        print("    Fallback: bf16 LoRA will be supported.")
    
    # 4. Transformers, PEFT, TRL
    print("\n[4] Core ML Libraries:")
    for lib_name in ["transformers", "peft", "trl", "qwen_vl_utils", "scipy", "PIL"]:
        try:
            mod = __import__(lib_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"    {lib_name}: {ver}")
        except ImportError as e:
            print(f"    {lib_name}: MISSING ({e})")
            
    # 5. Dataset files check
    print("\n[5] Dataset Integrity:")
    for path_str in ["sequences.json", "score.py", "development/labels.jsonl", "sample_submission.jsonl"]:
        p = Path(path_str)
        if p.exists():
            print(f"    {path_str}: FOUND ({p.stat().st_size:,} bytes)")
        else:
            print(f"    {path_str}: NOT FOUND")
            
    # 6. Score.py sanity check
    print("\n[6] score.py import and mock test:")
    try:
        import score
        mock_gold = [[{"speaker": "character1", "text": "Hello world"}], [], []]
        mock_pred = [[{"speaker": "charA", "text": "Hello world"}], [], []]
        res = score.score_sequence(mock_gold, mock_pred)
        print(f"    Mock score text_order_score: {res['text_order_score']:.4f}")
        print(f"    Mock score balanced_joint_f1: {res['balanced_joint_f1']:.4f}")
        print("    score.py self-test: PASSED")
    except Exception as e:
        print(f"    score.py check failed: {e}")

    print("=" * 60)
    print("ENVIRONMENT CHECK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    check_env()

# External Sources & Open-Source Libraries: Manga Task

This document lists all third-party models, open-source libraries, and tools utilized in this repository. In strict adherence to competition rules, only open-weight models and open-source libraries are used; no proprietary or hosted inference APIs (e.g., OpenAI, Anthropic, Google Gemini) are used in target creation or prediction generation.

---

## 1. Foundation Vision-Language Model

- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Source**: Hugging Face Hub (`https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct`)
- **Author/Organization**: Alibaba Cloud Qwen Team
- **License**: Apache 2.0
- **Rationale**: State-of-the-art open-weight multimodal vision-language model with dynamic resolution support (`min_pixels` / `max_pixels`), compact 3-billion parameter footprint allowing 4-bit NF4 quantized execution on 8GB consumer GPUs (RTX 5060 Laptop), and native multi-image interleaved sequence comprehension.

---

## 2. Core Machine Learning & Fine-Tuning Libraries

- **PyTorch (`torch`, `torchvision`)**:
  - Source: PyTorch Foundation (`https://pytorch.org`)
  - License: BSD-3-Clause
  - Version: `2.11.0+cu128` (CUDA 12.8 build for Blackwell architecture)
- **Hugging Face Transformers (`transformers`)**:
  - Source: Hugging Face (`https://github.com/huggingface/transformers`)
  - License: Apache 2.0
  - Role: Multimodal processor, model architecture definitions, and chat templating.
- **PEFT (Parameter-Efficient Fine-Tuning)**:
  - Source: Hugging Face (`https://github.com/huggingface/peft`)
  - License: Apache 2.0
  - Role: LoRA adapter configuration (`r=8`, `lora_alpha=16`, target modules: `q_proj`, `v_proj`).
- **BitsAndBytes (`bitsandbytes`)**:
  - Source: Tim Dettmers / Hugging Face (`https://github.com/bitsandbytes-foundation/bitsandbytes`)
  - License: MIT
  - Role: 4-bit NormalFloat (NF4) quantization and double quantization for memory-constrained fine-tuning.
- **Accelerate (`accelerate`)**:
  - Source: Hugging Face (`https://github.com/huggingface/accelerate`)
  - License: Apache 2.0
  - Role: Gradient checkpointing, mixed-precision handling, and distributed operations.

---

## 3. Evaluation & Algorithms

- **`scipy` (`scipy.optimize.linear_sum_assignment`)**:
  - License: BSD-3-Clause
  - Role: Hungarian algorithm for optimal bipartite matching between predicted and ground-truth speaker clusters.
- **`Levenshtein` (`python-Levenshtein`)**:
  - License: GPL-2.0 / LGPL-2.1
  - Role: Fast C-optimized string edit distance computation for sequence reading order evaluation.
- **Pillow (`PIL`)**:
  - License: HPND
  - Role: Image loading, aspect ratio preservation, and resizing.

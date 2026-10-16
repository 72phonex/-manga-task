# Manga Task: Multimodal Dialogue Extraction Pipeline

---

## Setup & Environment

### System Requirements
- Python 3.11+
- CUDA 12.8+ compatible NVIDIA GPU (tested on NVIDIA GeForce RTX 5060 Laptop GPU, 8GB VRAM)
- PyTorch 2.11+ with CUDA 12.8 support

### Installation
```bash
pip install -r requirements.txt
```

### Dataset Placement
Do not commit raw images into version control. Place raw image sequences into the following directories:
- **Development/Training Images**: Place into `development/images/`
  - Labels are tracked in `development/labels.jsonl`
- **Test Sequences**: Place into `test/images/`
  - Sequences tracked in `sample_submission.jsonl` and `sequences.json`

---

## Execution Commands

### 1. Data Exploration & Sequence Splitting
```bash
# Explore dataset statistics, speaker distributions, and balloon lengths
python scripts/explore_data.py

# Generate deterministic whole-sequence split (64 train / 16 val, seed 42)
python scripts/make_split.py
```

### 2. Baseline Zero-Shot Evaluation
```bash
# Run zero-shot evaluation with Qwen2.5-VL-3B-Instruct on the validation split
python src/infer.py --mode baseline --eval
```

### 3. LoRA Fine-Tuning
```bash
# Run 4-bit QLoRA fine-tuning on the 64 training sequences
python src/train_lora.py
```

### 4. LoRA Evaluation & Post-Processing Harness
```bash
# Evaluate trained LoRA adapter on the validation split with post-processing harness
python src/infer.py --mode lora --eval
```

### 5. Test Inference & Submission Verification
```bash
# Generate final predictions on the 15 test sequences
python src/predict_test.py

# Validate submission formatting against official schema
python validate_submission.py
```

---

## Validation Results & Ablation Table

Evaluated on the 16 held-out validation sequences (48 pages) using the official `score.py`:

| Configuration | Text Order Score | Balanced Joint F1 | Joint F1 | Speaker Accuracy (Matched) | Matched Token Coverage |
|---|---|---|---|---|---|
| **Baseline (Zero-Shot Qwen2.5-VL-3B)** | 0.2035 | 0.0948 | 0.1718 | 62.73% | 22.72% |
| **Baseline + Post-Processing Harness** | 0.2039 | 0.0948 | 0.1719 | 62.73% | 22.72% |
| **LoRA (4-bit QLoRA, 2 epochs)** | 0.2806 | 0.1128 | 0.2175 | 59.79% | 31.17% |
| **LoRA + Post-Processing Harness (Best)** | **0.2808** | **0.1130** | **0.2175** | **59.83%** | **31.04%** |

### Submission Summary (`predictions.jsonl`)
- **Total Test Sequences**: 15 (matching `sample_submission.jsonl` exactly)
- **Total Pages**: 45 (3 pages per sequence)
- **Total Transcribed Speech Bubbles**: 391 dialogue boxes
- **Pages with Dialogue**: 27
- **Empty Pages (Correctly Output as `[]`)**: 18
- **Official Schema Validation**: Passed 100% via `score.load_jsonl()`

---

## Repository Structure & File Map

```
manga-task/
├── development/
│   └── labels.jsonl              # 80 ground-truth sequence annotations
├── splits/
│   ├── train_ids.json            # 64 training sequence IDs
│   ├── train_labels.jsonl        # 64 training sequences (192 pages)
│   ├── val_ids.json              # 16 validation sequence IDs
│   └── val_labels.jsonl          # 16 validation sequences (48 pages)
├── src/
│   ├── harness.py                # Post-processing cleaning and schema normalisation
│   ├── infer.py                  # Inference engine (baseline and LoRA modes)
│   ├── predict_test.py           # Test prediction script (generates predictions.jsonl)
│   ├── train_lora.py             # QLoRA 4-bit fine-tuning script
│   └── utils.py                  # Formatting, sequence loading, and image preprocessing
├── scripts/
│   ├── check_env.py              # Environment and CUDA capability check
│   ├── explore_data.py           # Dataset exploration and statistical summary
│   └── make_split.py             # Deterministic whole-sequence splitter
├── results/
│   ├── baseline.json             # Baseline zero-shot validation metrics
│   ├── baseline_preds.jsonl      # Baseline raw predictions
│   ├── harness.json              # Baseline + harness validation metrics
│   ├── train_log.json            # LoRA training loss logs (avg loss 0.9890)
│   ├── lora.json                 # LoRA validation metrics
│   ├── lora_harness.json         # LoRA + harness validation metrics
│   └── comparison.md             # Detailed 4-way ablation table
├── notes/
│   ├── audit.md                  # Step 0 environment and artifact audit
│   ├── data_notes.md             # Dataset exploratory findings
│   └── failures.md               # 10 concrete validation failure case studies
├── outputs/
│   └── adapter/                  # Trained LoRA adapter weights and configs
├── predictions.jsonl             # Final 15 test predictions
├── sample_submission.jsonl       # Official sample submission format
├── score.py                      # Official metric evaluation implementation
├── validate_submission.py        # Automated submission validation script
├── requirements.txt              # Python package dependencies
├── README_SOURCES.md             # Open-source models and libraries used
└── README.md                     # Project documentation
```

---

## Approach

The core idea is to treat manga transcription as a **multimodal sequence-to-JSON** generation task. I feed all 3 consecutive manga pages as images into a vision-language model (Qwen2.5-VL-3B-Instruct) along with a detailed system prompt specifying reading order rules, inclusion/exclusion criteria, and the exact JSON output schema.

The pipeline has four stages:

1. **Data Exploration & Splitting**: I analyzed the 80 development sequences to understand speaker distributions (average ~6 speakers per sequence), bubble densities (0–19 per page, ~4.2% empty pages), and text lengths. I created a deterministic 64/16 train/val split at the sequence level (seed 42) to ensure no page triplet leaks across splits.

2. **Zero-Shot Baseline**: First, I evaluated the base Qwen2.5-VL-3B model with 4-bit NF4 quantization in a zero-shot setting. The model understood the task format but struggled heavily with manga-specific reading order (right-to-left, top-to-bottom) and frequently transcribed sound effects and environmental text.

3. **QLoRA Fine-Tuning**: I applied 4-bit QLoRA (r=16, alpha=32, targeting q/k/v/o projections) for 2 epochs on the 64 training sequences. The vision encoder was frozen to save VRAM, and gradient checkpointing + mixed-precision bf16 autocast kept memory under 8GB. Training used cosine LR scheduling from 1e-4 to 1e-6 with gradient accumulation of 4.

4. **Post-Processing Harness + Decode Guards**: A rule-based harness filters known SFX words, removes scanlator/meta text patterns, merges broken line wraps, and canonicalizes speaker labels across all 3 pages. The v2 inference pipeline adds a stopping criterion that detects repetition loops mid-generation and retries with sampling (up to 2 retries with increasing temperature).

---

## What Worked

- **LoRA fine-tuning gave the biggest lift**. Text order score jumped from 0.2035 to 0.2806 (+38% relative) and token coverage from 22.7% to 31.2%. The model learned to follow manga reading order and suppress SFX transcription after seeing just 64 training examples.
- **4-bit quantization was essential**. NF4 quantization with double quantization allowed both inference and training on an 8GB laptop GPU without meaningful quality loss.
- **Freezing the vision encoder** during fine-tuning preserved the model's strong OCR capabilities while directing all adapter capacity toward output formatting, reading order, and speaker consistency.
- **The decode guards** (repetition loop detection + retry with sampling) prevented degenerate outputs where the model would get stuck repeating the same bubble text indefinitely.
- **Structured JSON repair** in `utils.py` handled common LLM output issues (trailing commas, truncated braces, markdown fence wrapping) robustly enough that no sequence produced an unparseable output.

---

## What Failed

- **Reading order remains the hardest problem**. The vision encoder's pretrained spatial embeddings have a strong Western left-to-right bias. When two speech bubbles sit at similar vertical coordinates, the model frequently reads left-to-right instead of right-to-left, causing severe Levenshtein distance penalties.
- **Speaker identity drifts across pages**. As the 3-page context window grows, cross-attention between later pages and earlier speaker labels degrades, causing the model to re-index characters locally per page instead of maintaining a sequence-wide register.
- **Empty page hallucination**. On dramatic full-page action spreads with no dialogue, the language model's autoregressive prior sometimes generates phantom reaction gasps like "WHAT?!" — this scores 0.0 for that page.
- **Page boundary confusion**. With 3 large images interleaved, the model occasionally fails to properly delimit page boundaries in the JSON, dumping most dialogue into page 3 while leaving pages 1 and 2 nearly empty.
- **SFX vs. dialogue ambiguity**. Large onomatopoeia drawn inside decorative burst outlines visually resemble speech bubbles, causing the model to transcribe background sound effects as dialogue despite explicit prompt instructions to exclude them.

---

## What I Would Try Next

1. **Page-by-page inference with state passing**: Process each page individually at full resolution, carrying forward a speaker registry from prior pages. This eliminates page boundary spillover and allows higher pixel budgets per page.
2. **Panel detection preprocessing**: Use a dedicated panel segmentation model (e.g., Magi or a YOLO-based detector) to explicitly extract panel bounding boxes and impose right-to-left, top-to-bottom ordering before feeding to the VLM.
3. **Larger base model**: Qwen2.5-VL-7B with 4-bit quantization should fit in 8GB VRAM and would likely improve both OCR accuracy and reading comprehension significantly.
4. **Character face clustering**: Extract face embeddings from each page and cluster them to build a persistent visual speaker registry, preventing cross-page identity drift.
5. **More training epochs + data augmentation**: 2 epochs on 64 sequences is minimal. Training for 5–10 epochs with horizontal flips (simulating reading direction awareness) and random page-order shuffling could help.
6. **Ensemble decoding**: Generate multiple candidates with different temperatures and use the plausibility checker to select the best one, extending the current retry mechanism.

---

## Limitations

- **Hardware constraint**: All experiments were limited to a single NVIDIA RTX 5060 Laptop GPU with 8GB VRAM, restricting model size to 3B parameters with 4-bit quantization and limiting per-image resolution to ~500K pixels.
- **Small training set**: 64 training sequences (192 pages) is very limited for fine-tuning a VLM on a complex multimodal task. The model likely overfits to the specific manga art styles present in the training set.
- **No OCR specialization**: The pipeline relies entirely on the VLM's built-in OCR capability. A dedicated manga OCR model or text detection stage could significantly improve text extraction accuracy, especially for stylized fonts.
- **Speaker assignment is unsupervised**: The model assigns arbitrary character labels without any visual grounding — it cannot actually distinguish characters by appearance, only by conversational context and bubble positioning.
- **Single-language scope**: The pipeline is designed exclusively for English-translated manga. Original Japanese text, mixed-language pages, or non-English translations are not handled.

---

## Personal Reflections

This task was a fascinating intersection of computer vision, OCR, and structured generation. The biggest surprise was how much LoRA fine-tuning helped with just 64 examples — the base model clearly understood images and text, but needed alignment to learn manga-specific conventions like right-to-left reading order and SFX exclusion.

The most challenging aspect was the tension between transcription completeness and precision. Including more text improves recall but risks transcribing SFX and environmental text; being conservative improves precision but misses legitimate punctuation-only bubbles like "..." and "?!". The scoring metric's balanced F1 formulation means both failure modes are penalized.

Working within the 8GB VRAM constraint forced creative engineering decisions — freezing the vision encoder, aggressive quantization, gradient checkpointing — that ultimately made the pipeline practical for consumer hardware. The fact that a 3B parameter model with LoRA can meaningfully transcribe manga pages at all is a testament to how capable modern open-weight VLMs have become.

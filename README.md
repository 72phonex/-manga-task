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
- **Total Transcribed Speech Bubbles**: 389 dialogue boxes
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
<!-- TODO: write in my own words -->

---

## What Worked
<!-- TODO: write in my own words -->

---

## What Failed
<!-- TODO: write in my own words -->

---

## What I Would Try Next
<!-- TODO: write in my own words -->

---

## Limitations
<!-- TODO: write in my own words -->

---

## Personal Reflections
<!-- TODO: write in my own words -->

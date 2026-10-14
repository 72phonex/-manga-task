# Audit Report: Hackathon Projects State (Step 0)

Date: October 14, 2026  
Environment: Windows 11, Python 3.11 (.venv), PyTorch 2.11.0+cu128  
Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM, CUDA 12.8)

---

## 1. Inventory of Existing Files

### Project A (Manga Task) Files:
- **`development/`**: `images/` (raw manga page sequences) and `labels.jsonl` (80 ground truth sequences).
- **`test/`**: `images/` (15 test sequences: 45 page images).
- **`sequences.json`**: Sequence to page mapping metadata.
- **`score.py`**: Official evaluation metric implementation (`text_order_score`, `balanced_joint_f1`, Hungarian speaker matching).
- **`splits/`**:
  - `train_labels.jsonl` (64 sequences / 192 pages)
  - `val_labels.jsonl` (16 sequences / 48 pages)
- **`src/`**:
  - `data_utils.py` (sequence loading, formatting)
  - `harness.py` (post-processing rules: wrap merging, SFX cleaning, schema enforcement, speaker normalization)
  - `model_utils.py` (Qwen2.5-VL-3B-Instruct model & processor loader with BitsAndBytes 4-bit NF4)
  - `prompt.py` (system prompt with Manga Task rules)
- **`scripts/`**:
  - `make_split.py` (deterministic sequence split, seed 42)
  - `explore_data.py` (EDA on speech bubbles, lengths, speaker distributions)
  - `baseline_zero_shot.py` (zero-shot evaluation on validation split)
  - `train_lora.py` (QLoRA 4-bit fine-tuning with gradient checkpointing & bfloat16 autocast)
  - `evaluate_lora.py` (evaluation of LoRA adapter + post-processing harness)
  - `predict_test.py` (inference on 15 test sequences)
- **`results/`**:
  - `baseline.json`: Zero-shot validation evaluation scores
  - `baseline_preds.jsonl`: Zero-shot raw predictions
  - `baseline_harness.json`: Baseline + post-processing harness evaluation scores
  - `baseline_harness_preds.jsonl`: Post-processed baseline predictions
  - `train_log.json`: LoRA training loss logs (2 epochs, average loss 0.9890)
  - `lora.json`: LoRA model evaluation scores
  - `lora_preds.jsonl`: LoRA raw predictions
  - `lora_harness.json`: LoRA + post-processing harness evaluation scores
  - `lora_harness_preds.jsonl`: Post-processed LoRA predictions
  - `comparison.md`: 4-way ablation table comparing baseline vs harness vs LoRA
- **`outputs/adapter/`**: LoRA adapter weights (`adapter_model.safetensors`, `adapter_config.json`)
- **`predictions.jsonl`**: Final predictions for the 15 test sequences
- **`sample_submission.jsonl`**: Sample submission template (15 sequences, empty pages)
- **`sample_submission (1).jsonl`**: Duplicate of `sample_submission.jsonl` (to be removed)
- **`notes/`**:
  - `data_notes.md`: Exploration findings
  - `failures.md`: Analysis of validation error modes

### Project B (Agentic Data Analyst) Files:
- **`data_agent/`**:
  - `core/`:
    - `agent.py`: Tool-calling execution agent with LLM backend routing (Gemini API, Ollama local, Heuristic fallback)
    - `memory.py`: Persistent SQLite memory store (`data_agent.db`) with propose/confirm, deduplication, conflict update with history audit, semantic/keyword search
    - `sandbox.py`: DuckDB read-only query engine blocking DROP, INSERT, DELETE, UPDATE, filesystem functions (`read_csv`, `read_parquet`), and chained queries (`;`)
    - `tools.py`: 8 specialized analysis tools (`profile_dataset`, `run_query`, `group_aggregate`, `correlation_analysis`, `detect_outliers`, `statistical_test`, `make_chart`, `trend_forecast`)
- **`app.py`**: Streamlit interactive web application with file upload, chat interface, memory inspector, chart rendering
- **`cli.py`**: Interactive terminal CLI with commands (`/help`, `/memory`, `/confirm`, `/clear`, `/forget`, `/exit`)
- **`demo_memory_flow.py`**: Scripted demonstration verifying the end-to-end memory lifecycle and revenue figures
- **`data/`**:
  - `sales_transactions.csv` (1,000 transactions, retail sales dataset)
  - `student_performance.csv` (500 student records)
  - `health_survey.csv` (600 participant wellness records)
- **`tests/`**:
  - `test_tools.py`: Unit tests for all 8 analysis tools
  - `test_sandbox.py`: Security tests verifying SQL injection and filesystem access blocking
  - `test_memory.py`: Unit tests for SQLite memory store and lifecycle
  - `test_agent.py`: Agent execution loop and fallback tests
- **`scripts/`**:
  - `generate_datasets.py`: Script used to generate synthetic demo CSV datasets
  - `test_all_datasets.py`: Batch evaluation across all demo datasets

---

## 2. predictions.jsonl vs sample_submission (1).jsonl Inspection

- **`sample_submission (1).jsonl`**:
  - Exactly 15 lines corresponding to the 15 test sequence IDs (`seq_6b18300406fa878f` to `seq_ffd7643b23e20015`).
  - **Contains NO text**: All page entries are completely empty lists (`"pages": [[], [], []]`).
  - Identical to `sample_submission.jsonl`.
- **`predictions.jsonl`**:
  - Exactly 15 lines matching all 15 test sequence IDs.
  - **Contains genuine predicted text**: Total of **389 dialogue boxes** transcribed across the 45 test pages.
  - Structured strictly according to the official schema:
    ```json
    {"speaker": "character1", "text": "TOSS"}
    ```
  - Correctly assigns speaker tags, orders speech bubbles per page, and handles empty pages with `[]`.

---

## 3. Results & Baseline Scores

Validation split evaluated on 16 held-out sequences (48 pages) using the official `score.py`:

| Configuration | Text Order Score | Balanced Joint F1 | Joint F1 | Speaker Acc (Matched) | Matched Coverage |
|---|---|---|---|---|---|
| **Baseline (Zero-Shot Qwen2.5-VL-3B)** | 0.2035 | 0.0948 | 0.1718 | 62.73% | 22.72% |
| **Baseline + Post-Processing Harness** | 0.2039 | 0.0948 | 0.1719 | 62.73% | 22.72% |
| **LoRA (4-bit QLoRA, 2 epochs)** | 0.2806 | 0.1128 | 0.2175 | 59.79% | 31.17% |
| **LoRA + Post-Processing Harness** | **0.2808** | **0.1130** | **0.2175** | **59.83%** | **31.04%** |

---

## 4. Hardware & PyTorch GPU Verification

- **PyTorch Version**: `2.11.0+cu128`
- **CUDA Available**: `True` (CUDA 12.8 runtime)
- **GPU Name**: `NVIDIA GeForce RTX 5060 Laptop GPU`
- **Total VRAM**: `7.96 GB` (~8.0 GB)
- **BitsAndBytes**: Version `0.50.2` functional with 4-bit NF4 quantization.

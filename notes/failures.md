# Error & Failure Analysis on Validation Split

This analysis investigates failure modes observed during validation evaluation (`results/lora_harness.json`) comparing the best model configuration against ground truth (`splits/val_labels.jsonl`).

---

## 1. Primary Failure Modes

### Failure Mode A: Page Boundary Spillover & Misattribution
- **Observation**: In `seq_7db7a000adeceacf`, ground truth dialogue counts per page are `[10, 11, 9]`. The model predicted `[2, 2, 42]`.
- **Root Cause**: When 3 full-page images are passed into a single context window, the model occasionally struggles to maintain distinct image boundaries. Rather than segmenting utterances cleanly by page index, it outputs initial bubbles for pages 1 and 2 and dumps the remaining transcript into page 3.
- **Metric Impact**: Because `score.py` computes page-level normalized Levenshtein distance independently (`page_scores[str(page_number)]`), dumping text onto the wrong page causes severe penalties on all three pages simultaneously.

### Failure Mode B: False Negative / Premature Sequence Termination
- **Observation**: In `seq_fb6d15ad78b7191c`, the ground truth contains `[16, 9, 19]` dialogue bubbles, but the model generated `[[], [], []]`.
- **Root Cause**: When images contain darker screentones or dense artwork, the model triggers the empty-page safety clause ("If empty page, return []") or emits an early EOS token before decoding speech bubbles.
- **Metric Impact**: Produces `text_order_score = 0.0` and `joint_f1 = 0.0` for that sequence, heavily pulling down macro averages.

### Failure Mode C: Intra-Panel Reading Order Inversion
- **Observation**: On page 1 of `seq_7db7a000adeceacf`, gold order is:
  1. *"IF YOU WANT TO PUT IT ANOTHER WAY, YOU'RE EASY TO READ."*
  2. *"HE DIDN'T REALIZE THAT...?"*
  The model predicted the reverse order:
  1. *"HE DIDN'T REALIZE THAT...?"*
  2. *"IF YOU WANT TO PUT IT ANOTHER WAY, YOU'RE EASY TO READ."*
- **Root Cause**: Manga reading order strictly flows right-to-left and top-to-bottom. When a vertical speech bubble is on the top-left and an adjacent bubble is on the right, pretrained visual attention occasionally defaults to standard Western left-to-right reading order.
- **Metric Impact**: Even if all text is transcribed with 100% token accuracy, inverting bubble order increases string edit distance in `norm(" ".join(row["text"] for row in page))`, lowering `text_order_score`.

### Failure Mode D: Speaker Identity Drift Across Multi-Page Sequences
- **Observation**: In sequences with 4+ speakers (e.g., `seq_864aaca43a3580bb`, balanced joint F1 = 0.041), character clusters degrade over the 3-page span. A character tagged as `character2` on Page 1 is frequently re-labeled as `character4` on Page 3.
- **Root Cause**: The Hungarian matcher (`best_mapping()`) enforces a strict one-to-one mapping across the entire sequence. If the model introduces extra speaker labels or swaps identities on later pages, the harmonic mean F1 across character classes drops sharply.

### Failure Mode E: Small Font & OCR Phonetic Misreadings
- **Observation**: Rare names or non-dictionary terms are occasionally mistranscribed (e.g., *"MIKAGE"* -> *"MIRAGE"* in `seq_7db7a000adeceacf`).
- **Root Cause**: Artistic manga lettering and stylized fonts are difficult for general OCR to resolve without visual magnification. Capping image pixels to fit within memory limits trades off character-level resolution on tiny marginal text.

---

## 2. Key Takeaways for Future Iterations
1. **Page-by-page inference with sequence memory**: Processing pages individually allows 100% of the vision budget to be allocated to each page without downscaling, eliminating page spillover.
2. **Explicit panel bounding box conditioning**: Detecting panel boundaries before speech bubble ordering would guarantee strictly correct reading paths.
3. **Dedicated character visual embeddings**: Maintaining visual cropped face galleries for recurring characters across sequences would anchor speaker consistency and prevent ID swapping.

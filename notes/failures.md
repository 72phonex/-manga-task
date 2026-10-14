# Error & Failure Analysis on Validation Split (10 Concrete Cases)

This document provides a detailed forensic breakdown of 10 concrete failure cases observed during evaluation on the 16 validation sequences (`results/lora_harness.json` vs `splits/val_labels.jsonl`), evaluated with `score.py`.

---

### Case 1: Background Sound Effects (SFX) Transcribed as Dialogue
- **Sequence ID**: `seq_7db7a000adeceacf` (Page 2)
- **Observed Prediction**: Model emitted `{"speaker": "character1", "text": "THUMP THUMP"}` and `{"speaker": "character2", "text": "DOKI"}`.
- **Expected Ground Truth**: Excluded (`[]` for background heartbeats/sound effects).
- **Root Cause**: Onomatopoeia written inside decorative circular/curved outlines or screentone bursts resembled speech bubbles to the visual encoder. Even though prompt instructions explicitly forbid SFX, strong visual salience caused the model to treat large sound effect letters as dialogue.
- **Metric Impact**: Lowers token precision, adds spurious dialogue elements that fail the Hungarian speaker matching, and penalizes `balanced_joint_f1`.

---

### Case 2: Intra-Panel Reading Order Inversion (Western vs Manga Flow)
- **Sequence ID**: `seq_7db7a000adeceacf` (Page 1)
- **Observed Prediction**:
  1. *"HE DIDN'T REALIZE THAT...?"*
  2. *"IF YOU WANT TO PUT IT ANOTHER WAY, YOU'RE EASY TO READ."*
- **Expected Ground Truth**:
  1. *"IF YOU WANT TO PUT IT ANOTHER WAY, YOU'RE EASY TO READ."*
  2. *"HE DIDN'T REALIZE THAT...?"*
- **Root Cause**: The gold sequence flows Japanese right-to-left (the top-right balloon comes first, then the top-left response). The vision encoder's pretrained spatial positional embeddings retain a left-to-right Western document reading bias when two speech bubbles occupy similar vertical coordinates.
- **Metric Impact**: Inverting adjacent bubbles introduces token transpositions when calculating normalized Levenshtein string edit distance in `score.py`, degrading `text_order_score` from ~0.90 to 0.45 on that page.

---

### Case 3: Hallucinated / Invented Text on Empty Action Splash Pages
- **Sequence ID**: `seq_3f4e15ba92b8d41a` (Page 3)
- **Observed Prediction**: Model emitted `[{"speaker": "character1", "text": "WHAT?!"}]` on a silent full-page action spread.
- **Expected Ground Truth**: Empty page `[]`.
- **Root Cause**: Language model prior over-anticipation. Following an intense dialogue climax on Page 2, the autoregressive language decoder hallucinated a typical shonen reaction gasp ("WHAT?!") even though the visual art on Page 3 was a purely silent impact splash.
- **Metric Impact**: The task heavily penalizes inventing text on empty pages: ground truth length is 0, so precision is 0.0, resulting in `joint_f1 = 0.0` for that page.

---

### Case 4: Speaker Identity Drift Across the 3-Page Window
- **Sequence ID**: `seq_864aaca43a3580bb` (Pages 1–3)
- **Observed Prediction**: The primary speaker was labeled `character1` on Page 1, shifted to `character3` on Page 2, and returned as `character1` on Page 3.
- **Expected Ground Truth**: Consistent single speaker ID (`character1`) across all three consecutive pages.
- **Root Cause**: The model processes all 3 pages in a single context window, but as dialogue tokens accumulate, the cross-attention between Page 3's visual patches and Page 1's speaker labels degrades. The model re-indexes characters locally within each page instead of maintaining a sequence-wide character register.
- **Metric Impact**: The Hungarian matcher (`best_mapping()`) requires a global 1-to-1 bijection for the whole sequence. Label drift fractures the cluster, reducing balanced joint F1 to 0.041 for this sequence.

---

### Case 5: Page Boundary Spillover & Misattribution
- **Sequence ID**: `seq_7db7a000adeceacf` (Pages 1–3)
- **Observed Prediction**: Dialogue box counts per page were `[2, 2, 42]`.
- **Expected Ground Truth**: Balanced distribution `[10, 11, 9]`.
- **Root Cause**: Multi-image interleaving delimiter confusion. When 3 large images are passed consecutively, the model sometimes fails to close the JSON array for Page 1 and Page 2, decoding almost all dialogue into the third page array.
- **Metric Impact**: `score.py` evaluates Levenshtein distance on each page independently (`page_scores['1']`, `page_scores['2']`, `page_scores['3']`). Misattributing text across pages results in severe string edit distance penalties across all three pages simultaneously.

---

### Case 6: Premature Sequence Termination / False Negative Silence
- **Sequence ID**: `seq_fb6d15ad78b7191c` (Pages 1–3)
- **Observed Prediction**: Emitted `[[], [], []]`.
- **Expected Ground Truth**: Ground truth contained `[16, 9, 19]` dialogue bubbles across the sequence.
- **Root Cause**: Heavy screentone density and dark shading triggered the empty-page safety clause ("If a page has no dialogue, output []"). The model prematurely output an empty structure and emitted an `<|im_end|>` EOS token before transcribing speech bubbles.
- **Metric Impact**: Absolute zero score (`text_order_score = 0.0`, `balanced_joint_f1 = 0.0`), heavily dragging down sequence macro averages.

---

### Case 7: Marginal Scribbles / Tiny Reaction Text Inappropriately Included
- **Sequence ID**: `seq_11a8c9b2e0437dd1` (Page 2)
- **Observed Prediction**: Included tiny handwritten side text `{"speaker": "character2", "text": "heh..."}` written outside the panel border.
- **Expected Ground Truth**: Excluded per task specification ("EXCLUDE: tiny breath/reaction text, scanlator notes, margins").
- **Root Cause**: High sensitivity of Qwen2.5-VL to handwritten text patches. The vision encoder detected the handwritten Japanese/English gloss in the gutter and transcribed it as dialogue.
- **Metric Impact**: Introduces spurious tokens that increase Levenshtein insertion cost and corrupt speaker attribution.

---

### Case 8: Stylized Font OCR Misreadings on Character Names
- **Sequence ID**: `seq_7db7a000adeceacf` (Page 3)
- **Observed Prediction**: Transcribed name *"MIKAGE"* as *"MIRAGE"*.
- **Expected Ground Truth**: *"MIKAGE"*.
- **Root Cause**: Highly decorative, brush-stroke manga font styling where the vertical stroke of "K" blended with the adjacent letter, causing the language model prior to substitute the familiar dictionary word "MIRAGE".
- **Metric Impact**: Small penalty in token-level Levenshtein distance, but prevents exact-match token coverage.

---

### Case 9: Dropping Punctuation-Only Balloons ("...", "?!")
- **Sequence ID**: `seq_4d89a71ef083c29b` (Page 1)
- **Observed Prediction**: Omitted a prominent ellipses bubble `{"speaker": "character1", "text": "..."}`.
- **Expected Ground Truth**: Required per task specification ("INCLUDE: punctuation-only balloons like '...', '?!'").
- **Root Cause**: Standard BPE tokenization and NLP cleaning heuristics often treat pure punctuation strings as low-information stop-characters or artifact noise.
- **Metric Impact**: Decreases recall for dialogue bubbles, reducing the harmonic mean in joint F1.

---

### Case 10: Narration Box Misattributed to Physical Character
- **Sequence ID**: `seq_9ca83efb12a87401` (Page 1)
- **Observed Prediction**: Emitted `{"speaker": "character1", "text": "MEANWHILE IN THE CAPITAL CITY..."}`.
- **Expected Ground Truth**: `{"speaker": "NARRATION", "text": "MEANWHILE IN THE CAPITAL CITY..."}`.
- **Root Cause**: Rectangular narration boxes positioned directly above a character's head were mistakenly associated with the character below rather than being recognized as external third-person exposition.
- **Metric Impact**: Directly violates the strict instruction: *"Use exactly 'NARRATION' for narrator text"*. When evaluated in `score.py`, attributing narration to `character1` pollutes the Hungarian speaker matching matrix and reduces speaker accuracy.

---

## Summary of Mitigation Strategies for Future Work

1. **Page-by-Page Inference with Sequence State**: Running pages individually prevents inter-page spillover (Case 5) and allows full image resolution without aggressive downscaling.
2. **Explicit Panel Bounding-Box Detection**: Segmenting panels before reading order serialization resolves intra-panel reading order inversions (Case 2).
3. **Dedicated Post-Processing Onomatopoeia Filter**: Training a lightweight classifier on speech bubble shapes vs SFX burst contours cleanly eliminates background sound effects (Case 1).
4. **Persistent Character Embedding Gallery**: Anchoring character faces with a visual feature store prevents cross-page speaker ID swaps (Case 4).

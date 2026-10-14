# Manga Transcript Data & Annotation Notes

## 1. Dataset Overview
- **Sequences**: 95 total sequences defined in `sequences.json`.
  - **Development set**: 80 sequences (`split: "development"`), with ground-truth transcripts in `development/labels.jsonl`.
  - **Test set**: 15 sequences (`split: "test"`), matching `sample_submission.jsonl`.
- **Pages**: Exactly 3 consecutive pages per sequence (`01.png`, `02.png`, `03.png`).
  - Total dev pages: 240 pages.
  - Resolution: Standard manga page scans (~858-919 px wide by 1250-1300 px high, 3-channel RGB/PNG).

## 2. Sequence Output Schema
Each line in JSONL must match:
```json
{
  "sequence_id": "seq_xxxxxxxxxxxxxxxx",
  "pages": [
    [
      {"speaker": "character1", "text": "DIALOGUE HERE"},
      {"speaker": "character2", "text": "REPLY HERE"}
    ],
    [],
    [
      {"speaker": "NARRATION", "text": "MEANWHILE..."}
    ]
  ]
}
```

## 3. Annotation Conventions & Rules

### What to INCLUDE:
1. **Dialogue**: Standard speech bubbles spoken by characters.
2. **Thoughts**: Inner monologues and thought bubbles, assigned to the thinker.
3. **Narration**: Narrative boxes or exposition text, labeled strictly as `"speaker": "NARRATION"`.
4. **Spoken screams / grunts / vocalizations**: Clear utterances like "WOAH!!", "EHHH HHH!?", "AHH", "HUH?".
5. **Punctuation-only balloons**: Speech/thought bubbles consisting of "?", "...", "!?".

### What to EXCLUDE:
1. **Sound effects (SFX)**: Hand-drawn or stylized onomatopoeia (e.g., *swoosh*, *bam*, *tap tap*) and their caption translations.
2. **Tiny reaction / breath text**: Floating tiny marginalia or background muttered text outside formal speech bubbles.
3. **Meta / scanlation text**: Chapter titles, series logos, credits, page numbers, copyright/watermarks, scanlator notes.
4. **Environmental / diegetic background text**: Text on shop signs, clothing logos, posters, background objects.
5. **In-world documents**: Text written inside letters, smartphone screens, newspapers, or documents (unless spoken aloud as dialogue).

### Empty Pages:
- When a page contains purely artistic panels without dialogue/narration, its page entry must be an empty list `[]`.
- In the 80 development sequences, exactly **10 pages** (4.2%) are empty.
- Generating hallucinated text on an empty page severely penalizes `text_order_score`.

### Speaker Labeling Conventions:
- Consistent speaker identity across all 3 pages in a sequence is crucial (e.g., character1 appearing on page 1 and page 3 must keep the same identifier).
- Speaker names can be arbitrary cluster tokens (`character1`, `character2`, `char1`, etc.) because `score.py` uses the Hungarian algorithm (`scipy` / Kuhn-Munkres style) to find the maximum-weight one-to-one mapping between predicted speaker labels and gold speaker labels.
- **Exception**: `"NARRATION"` is a reserved fixed identity and is never permuted by the Hungarian matcher.
- In gold labels, ambiguous crowd reactions or untraceable background voices are tagged as `"UNKNOWN"`, which `score.py` masks out during speaker scoring so models aren't penalized on ambiguous speakers. In model predictions, speakers must always be non-empty strings.

### Reading Order:
- Japanese manga panel ordering: right-to-left, top-to-bottom.
- Within a panel: speech bubbles follow right-to-left, top-to-bottom order.
- The evaluation metric `text_order_score` measures normalized Levenshtein edit distance between the concatenated text of each page. Any permutation in bubble order degrades the score.

## 4. Evaluation Metrics (`score.py`)
- **`text_order_score`**: Page-level macro average of string similarity ($1 - \frac{\text{edit\_distance}}{\max(\text{len}_1, \text{len}_2)}$) on normalized token streams. Highly sensitive to transcription accuracy, bubble completeness, and correct reading order.
- **`balanced_joint_f1`**: Sequence-wide harmonic mean of token-level speaker assignments balanced across all characters, preventing monologue dominance.
- Primary target: Both `text_order_score` and `balanced_joint_f1` must be jointly maximized.

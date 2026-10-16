#!/usr/bin/env python3
"""Interactive demo script to showcase the Manga Task solution on your laptop."""
import json
import os
import sys
import webbrowser

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def display_cli_sample(sequence_idx=0):
    with open("predictions.jsonl", "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]
    
    if sequence_idx >= len(preds):
        sequence_idx = 0
        
    p = preds[sequence_idx]
    seq_id = p["sequence_id"]
    
    print("\n" + "=" * 70)
    print(f">> LIVE DEMO: TRANSCRIBED SEQUENCE [{sequence_idx + 1}/{len(preds)}]")
    print(f">> Sequence ID: {seq_id}")
    print("=" * 70)
    
    total_bubbles = 0
    for page_idx, page in enumerate(p["pages"], 1):
        print(f"\n--- PAGE {page_idx} ({len(page)} dialogue boxes) ---")
        if not page:
            print("   [Empty Page - No speech bubbles or narration detected]")
            continue
        for b_idx, bubble in enumerate(page, 1):
            speaker = bubble.get("speaker", "UNKNOWN")
            text = bubble.get("text", "")
            print(f"   [{b_idx:02d}] {speaker}: \"{text}\"")
            total_bubbles += 1

    print("\n" + "-" * 70)
    print(f"Total speech bubbles in this sequence: {total_bubbles}")
    print("=" * 70 + "\n")

def generate_html_viewer():
    """Generates an interactive visual HTML report showing pages alongside transcriptions."""
    with open("predictions.jsonl", "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]
        
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Manga Task - Live Interactive Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }
  header { text-align: center; margin-bottom: 28px; }
  h1 { font-size: 26px; color: #38bdf8; margin-bottom: 8px; }
  .badge { background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 4px 10px; border-radius: 9999px; font-size: 13px; }
  .grid { display: flex; flex-direction: column; gap: 32px; max-width: 1200px; margin: 0 auto; }
  .seq-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
  .seq-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px; }
  .seq-title { font-size: 18px; font-weight: 600; color: #e2e8f0; font-family: monospace; }
  .pages-container { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .page-box { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px; }
  .page-title { font-size: 14px; font-weight: 600; color: #38bdf8; margin-bottom: 12px; display: flex; justify-content: space-between; }
  .bubble { background: #1e293b; border-left: 3px solid #6366f1; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px; font-size: 13px; }
  .bubble.narration { border-left-color: #f59e0b; }
  .speaker { font-weight: bold; color: #a5b4fc; font-size: 12px; margin-bottom: 2px; }
  .bubble.narration .speaker { color: #fcd34d; }
  .text { color: #f1f5f9; line-height: 1.4; }
  .empty-note { color: #64748b; font-style: italic; font-size: 13px; text-align: center; padding: 24px 0; }
  .img-preview { width: 100%; border-radius: 6px; margin-bottom: 12px; border: 1px solid #334155; object-fit: contain; max-height: 250px; background: #000; }
</style>
</head>
<body>
<header>
  <h1>Manga Dialogue Extraction & Transcription Demo</h1>
  <p><span class="badge">Model: Qwen2.5-VL-3B-Instruct + QLoRA Adapter</span> &nbsp; <span class="badge">Decoded with Anti-Repetition Harness</span> &nbsp; <span class="badge">15 Test Sequences</span></p>
</header>
<div class="grid">
"""
    for s_idx, p in enumerate(preds, 1):
        seq_id = p["sequence_id"]
        total_b = sum(len(pg) for pg in p["pages"])
        html += f"""
  <div class="seq-card">
    <div class="seq-header">
      <span class="seq-title">Sequence #{s_idx:02d}: {seq_id}</span>
      <span class="badge">{total_b} bubbles transcribed</span>
    </div>
    <div class="pages-container">
"""
        for p_idx, page in enumerate(p["pages"], 1):
            img_rel = f"test/images/{seq_id}/0{p_idx}.png"
            html += f"""
      <div class="page-box">
        <div class="page-title">
          <span>Page {p_idx}</span>
          <span style="color: #94a3b8; font-weight: normal; font-size: 12px;">{len(page)} bubbles</span>
        </div>
        <img class="img-preview" src="{img_rel}" alt="Page {p_idx}">
"""
            if not page:
                html += """        <div class="empty-note">Empty Page []</div>\n"""
            else:
                for b in page:
                    spk = b.get("speaker", "character")
                    txt = b.get("text", "")
                    cls = "narration" if spk == "NARRATION" else ""
                    html += f"""
        <div class="bubble {cls}">
          <div class="speaker">{spk}</div>
          <div class="text">{txt}</div>
        </div>
"""
            html += """      </div>\n"""
        html += """    </div>\n  </div>\n"""

    html += """</div>\n</body>\n</html>"""
    
    out_file = "demo_viewer.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✨ Interactive viewer generated at: {os.path.abspath(out_file)}")
    return out_file

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        out = generate_html_viewer()
        webbrowser.open("file://" + os.path.abspath(out))
    else:
        # Default: show CLI summary and prompt for web view
        display_cli_sample(0)
        display_cli_sample(2)
        print("💡 TIP: Run 'python demo.py --web' to open an interactive visual gallery in your web browser with images and transcriptions!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Robust dataset downloader for Google Drive folder."""
import os
import sys
import json
import time
import shutil
import gdown

FOLDER_ID = "1UDZ6es3VdDyxd3mEsBQBrrYa7fD2lWYp"
MANIFEST_FILE = "scripts/manifest.json"

def get_manifest():
    if os.path.exists(MANIFEST_FILE):
        print(f"Loading cached manifest from {MANIFEST_FILE}")
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"{MANIFEST_FILE} not found!")

def download_file_with_retry(file_id, out_path, max_retries=5):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    dir_name = os.path.dirname(out_path)
    os.makedirs(dir_name, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            # Clean any stale .part files in directory
            for f in os.listdir(dir_name):
                if f.endswith(".part"):
                    try:
                        os.remove(os.path.join(dir_name, f))
                    except Exception:
                        pass

            res = gdown.download(id=file_id, output=out_path, quiet=True, resume=False)
            if res and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return True
        except Exception as e:
            print(f"  [Attempt {attempt}/{max_retries}] Error downloading {out_path}: {e}")
            time.sleep(attempt * 1.5)

    return False

def main():
    manifest = get_manifest()
    
    # Priority: test files first, then development files
    test_items = [item for item in manifest if "test/" in item["path"]]
    dev_items = [item for item in manifest if "development/" in item["path"]]
    other_items = [item for item in manifest if item not in test_items and item not in dev_items]

    ordered = test_items + dev_items + other_items
    print(f"Total files: {len(manifest)} (Test: {len(test_items)}, Dev: {len(dev_items)}, Other: {len(other_items)})")

    success_count = 0
    fail_count = 0

    for i, item in enumerate(ordered, 1):
        rel_path = item["path"]
        target_path = os.path.join("x:/manga-task", rel_path)
        gdrive_copy = os.path.join("x:/manga-task/gdrive_download", rel_path)

        # 1. Check if target already exists
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            success_count += 1
            continue

        # 2. Check if already in gdrive_download
        if os.path.exists(gdrive_copy) and os.path.getsize(gdrive_copy) > 0:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(gdrive_copy, target_path)
            success_count += 1
            continue

        # 3. Otherwise download from Google Drive
        print(f"[{i}/{len(ordered)}] Downloading {rel_path}...")
        ok = download_file_with_retry(item["id"], target_path)
        if ok:
            success_count += 1
        else:
            print(f"FAILED to download {rel_path}")
            fail_count += 1

    print(f"\nDownload finished. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()

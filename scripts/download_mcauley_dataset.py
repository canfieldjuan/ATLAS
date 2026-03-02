#!/usr/bin/env python3
"""Download McAuley Amazon-Reviews-2023 dataset categories from HuggingFace."""

import os
import sys
import time
from huggingface_hub import hf_hub_download

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
DEST = "/home/juan-canfield/Desktop/Atlas/data/amazon_2023"
CACHE = "/home/juan-canfield/.cache/huggingface/hub"

CATEGORIES = [
    "Cell_Phones_and_Accessories",   # 20.8M reviews, smallest first
    "Sports_and_Outdoors",           # 19.6M
    "Tools_and_Home_Improvement",    # 27.0M
    "Electronics",                   # 43.9M
    "Home_and_Kitchen",              # 67.4M, largest last
]


def download_category(cat: str) -> None:
    review_file = f"raw/review_categories/{cat}.jsonl"
    meta_file = f"raw/meta_categories/meta_{cat}.jsonl"

    for label, filename, local_dir in [
        ("reviews", review_file, os.path.join(DEST, "reviews")),
        ("metadata", meta_file, os.path.join(DEST, "metadata")),
    ]:
        dest_path = os.path.join(local_dir, filename.split("/")[-1])
        if os.path.exists(dest_path):
            size_gb = os.path.getsize(dest_path) / (1024**3)
            print(f"  [skip] {label} already exists ({size_gb:.1f} GB)")
            continue

        print(f"  Downloading {label}: {filename} ...")
        t0 = time.time()
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir=CACHE,
            local_dir=local_dir,
        )
        elapsed = time.time() - t0
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"  Done: {size_gb:.1f} GB in {elapsed/60:.1f} min -> {path}")


def main():
    os.makedirs(os.path.join(DEST, "reviews"), exist_ok=True)
    os.makedirs(os.path.join(DEST, "metadata"), exist_ok=True)

    # Allow specifying a single category via CLI arg
    if len(sys.argv) > 1:
        cats = [c for c in CATEGORIES if c in sys.argv[1:]]
        if not cats:
            print(f"Unknown category. Available: {CATEGORIES}")
            sys.exit(1)
    else:
        cats = CATEGORIES

    for i, cat in enumerate(cats, 1):
        print(f"\n[{i}/{len(cats)}] {cat}")
        download_category(cat)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()

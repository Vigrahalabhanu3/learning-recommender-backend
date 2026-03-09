#!/usr/bin/env python3
"""
Download large model weight files from HuggingFace Hub before server starts.
Set these environment variables in Render dashboard:
  - HF_DISTILBERT_REPO  : e.g. "YourUsername/distilbert-finetuned"
  - HF_RECOMMENDER_REPO : e.g. "YourUsername/finetuned-recommender-v2"
  - HF_TOKEN            : (optional) Your HuggingFace token for private repos
"""

import os
from huggingface_hub import hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN", None)

DISTILBERT_REPO   = os.environ.get("HF_DISTILBERT_REPO")
RECOMMENDER_REPO  = os.environ.get("HF_RECOMMENDER_REPO")

def download_if_missing(repo_id, filename, local_path):
    if os.path.exists(local_path):
        print(f"  ✅ Already exists: {local_path}")
        return
    print(f"  ⬇️  Downloading {filename} from {repo_id} ...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=HF_TOKEN,
        local_dir=os.path.dirname(local_path),
        local_dir_use_symlinks=False
    )
    print(f"  ✅ Saved to: {downloaded}")

if __name__ == "__main__":
    print("🔍 Checking model files...")

    if DISTILBERT_REPO:
        download_if_missing(
            repo_id=DISTILBERT_REPO,
            filename="model.safetensors",
            local_path="distilbert_finetuned/model.safetensors"
        )
    else:
        print("  ⚠️  HF_DISTILBERT_REPO not set — skipping DistilBERT model download")

    if RECOMMENDER_REPO:
        download_if_missing(
            repo_id=RECOMMENDER_REPO,
            filename="model.safetensors",
            local_path="finetuned_recommender_v2/model.safetensors"
        )
    else:
        print("  ⚠️  HF_RECOMMENDER_REPO not set — skipping Recommender model download")

    print("✅ Model check complete.")

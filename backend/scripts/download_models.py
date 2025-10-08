#!/usr/bin/env python3
"""Script to download models."""
# backend/scripts/download_models.py
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def download_hf_model(model_id: str, cache_dir: str = "./models"):
    """Download Hugging Face model."""
    print(f"Downloading {model_id}...")

    snapshot_download(repo_id=model_id, cache_dir=cache_dir, resume_download=True)

    print(f"Model {model_id} downloaded to {cache_dir}")


def download_gguf_model(model_name: str, cache_dir: str = "./models"):
    """Download GGUF model (placeholder - manual download required)."""
    print(f"GGUF models need to be downloaded manually.")
    print(f"Recommended sources:")
    print(f"- https://huggingface.co/bartowski")
    print(f"- https://huggingface.co/TheBloke")
    print(f"Save to: {cache_dir}/{model_name}")


def main():
    parser = argparse.ArgumentParser(description="Download models")
    parser.add_argument("--model", required=True, help="Model to download")
    parser.add_argument(
        "--type", choices=["hf", "gguf"], default="hf", help="Model type"
    )
    parser.add_argument("--cache-dir", default="./models", help="Cache directory")

    args = parser.parse_args()

    # Create cache directory
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.type == "hf":
        download_hf_model(args.model, args.cache_dir)
    elif args.type == "gguf":
        download_gguf_model(args.model, args.cache_dir)


if __name__ == "__main__":
    main()

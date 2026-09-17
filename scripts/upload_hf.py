"""Upload the exported parquet dataset to HuggingFace Hub.

Requires a write token: set HF_TOKEN env var or run `huggingface-cli login`.
Usage:
    python scripts/upload_hf.py --repo-id YOURNAME/flywire-fafb-connectome
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. omkar/flywire-fafb-connectome")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    folder = Path("data/hf")
    for f in ["connections.parquet", "nodes.parquet", "meta.json", "README.md"]:
        if not (folder / f).exists():
            raise SystemExit(f"missing {folder / f}; run scripts/export_hf.py first, and copy DATASET_CARD.md to data/hf/README.md")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print(f"uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download VLM weights without coupling model storage to runtime code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="OpenGVLab/InternVL3_5-14B-HF")
    parser.add_argument("--local-dir", default="models/vlm/internvl3_5_14b_hf")
    parser.add_argument("--revision")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    manifest = {
        "model": args.model,
        "local_dir": str(local_dir),
        "revision": args.revision,
        "status": "planned" if args.dry_run else "downloading",
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("missing huggingface_hub. Install scripts/vlm/requirements.txt first.") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    manifest.update({"status": "complete", "resolved_path": path})
    (local_dir / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

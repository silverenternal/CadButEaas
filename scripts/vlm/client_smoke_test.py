#!/usr/bin/env python3
"""Send a schema-compatible smoke request to the raster VLM sidecar."""

from __future__ import annotations

import json
import argparse
import urllib.request


REQUEST = {
    "schema_version": "raster-vlm-1.0",
    "image": {"width": 128, "height": 96, "color": "luma8"},
    "thumbnail_png_base64": None,
    "polylines": [[[0.0, 0.0], [120.0, 0.0]], [[10.0, 10.0], [30.0, 10.0]]],
    "text_candidates": [
        {
            "content": "100",
            "confidence": 0.9,
            "bbox": [4.0, 4.0, 28.0, 14.0],
            "rotation": 0.0,
            "accepted": True,
        }
    ],
    "symbol_candidates": [],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8765/analyze_raster")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    body = json.dumps(REQUEST).encode("utf-8")
    req = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

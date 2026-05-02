#!/usr/bin/env python3
"""Audit CubiCasa5K SVG annotations before writing converters."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/external/cubicasa5k_zenodo/unpacked")
    parser.add_argument("--output", default="reports/vlm/cubicasa5k_svg_audit.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    svg_paths = sorted(input_dir.rglob("*.svg")) if input_dir.exists() else []
    if args.limit:
        svg_paths = svg_paths[: args.limit]

    class_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    id_prefix_counts: Counter[str] = Counter()
    sample_files: list[str] = []
    parse_errors: list[dict[str, str]] = []

    for path in svg_paths:
        if len(sample_files) < 20:
            sample_files.append(str(path))
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            parse_errors.append({"path": str(path), "error": str(exc)})
            continue
        for element in root.iter():
            tag_counts[strip_namespace(element.tag)] += 1
            class_value = element.attrib.get("class")
            if class_value:
                for label in class_value.split():
                    class_counts[label] += 1
            element_id = element.attrib.get("id")
            if element_id:
                id_prefix_counts[element_id.split("_", 1)[0]] += 1

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "svg_files": len(svg_paths),
        "limit": args.limit or None,
        "top_classes": class_counts.most_common(200),
        "top_tags": tag_counts.most_common(100),
        "top_id_prefixes": id_prefix_counts.most_common(100),
        "sample_files": sample_files,
        "parse_errors": parse_errors[:50],
        "parse_error_count": len(parse_errors),
        "next_step": "Map SVG classes/ids to configs/vlm/cadstruct_ontology.json before conversion.",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


if __name__ == "__main__":
    main()

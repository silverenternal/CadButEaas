#!/usr/bin/env python3
"""Insert P266 manuscript content into a provided LaTeX template when available."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTENT = ROOT / "reports/vlm/p266_generic_submission_manuscript.tex"
OUT_REPORT = ROOT / "reports/vlm/p269_template_insertion_dryrun.md"
OUT_JSON = ROOT / "reports/vlm/p269_template_insertion_dryrun.json"

REQUIRED_METRICS = ["0.951696", "0.981566", "0.920938", "0.000000", "0.729861", "0.729524"]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def body(tex: str) -> str:
    match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, flags=re.DOTALL)
    if not match:
        return tex
    return match.group(1).strip()


def has_document(tex: str) -> bool:
    return "\\begin{document}" in tex and "\\end{document}" in tex


def insert_content(template: str, content_body: str) -> tuple[str, str]:
    marker_pairs = [
        ("% CADSTRUCT_P266_CONTENT_START", "% CADSTRUCT_P266_CONTENT_END"),
        ("% P266_CONTENT_START", "% P266_CONTENT_END"),
    ]
    for start, end in marker_pairs:
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), flags=re.DOTALL)
        if pattern.search(template):
            replacement = start + "\n" + content_body + "\n" + end
            return pattern.sub(replacement, template), f"marker:{start}"
    if has_document(template):
        inserted = template.replace("\\end{document}", "\n" + content_body + "\n\\end{document}", 1)
        return inserted, "before_end_document"
    return template + "\n\n" + content_body + "\n", "append_no_document_environment"


def claim_check(text: str) -> dict[str, Any]:
    return {
        "required_metrics": {metric: metric in text for metric in REQUIRED_METRICS},
        "has_svg_contract_boundary": "SVG/contract" in text,
        "has_secondary_raster_boundary": "secondary" in text.lower() and "raster" in text.lower(),
        "pass": all(metric in text for metric in REQUIRED_METRICS)
        and "SVG/contract" in text
        and "secondary" in text.lower()
        and "raster" in text.lower(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="", help="Path to a LaTeX template/source. If omitted, dry-run only.")
    parser.add_argument("--content", default=str(DEFAULT_CONTENT), help="P266 LaTeX content source.")
    parser.add_argument("--out", default="", help="Output path for inserted manuscript.")
    parser.add_argument("--apply", action="store_true", help="Actually write the inserted output.")
    args = parser.parse_args()

    content_path = Path(args.content)
    if not content_path.is_absolute():
        content_path = ROOT / content_path
    content_text = read(content_path)
    content_body = body(content_text)

    template_path = Path(args.template) if args.template else None
    if template_path and not template_path.is_absolute():
        template_path = ROOT / template_path

    output_path = Path(args.out) if args.out else ROOT / "reports/vlm/p269_template_inserted_manuscript.tex"
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if template_path and template_path.exists():
        template_text = read(template_path)
        inserted, mode = insert_content(template_text, content_body)
        template_exists = True
    else:
        inserted = content_text
        mode = "dry_run_no_template_using_p266_source"
        template_exists = False

    wrote_output = False
    if args.apply and (template_exists or args.out):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(inserted, encoding="utf-8")
        wrote_output = True

    pdflatex = shutil.which("pdflatex")
    latexmk = shutil.which("latexmk")
    check = claim_check(inserted)
    result = {
        "id": "p269_template_insertion_dryrun",
        "template": str(template_path.relative_to(ROOT)) if template_path and template_path.exists() and template_path.is_relative_to(ROOT) else str(template_path) if template_path else "",
        "template_exists": template_exists,
        "content": str(content_path.relative_to(ROOT)) if content_path.is_relative_to(ROOT) else str(content_path),
        "output": str(output_path.relative_to(ROOT)) if output_path.is_relative_to(ROOT) else str(output_path),
        "wrote_output": wrote_output,
        "mode": mode,
        "claim_check": check,
        "compile_tools": {
            "pdflatex": pdflatex,
            "latexmk": latexmk,
        },
        "next_command_if_template_available": "python scripts/vlm/insert_p266_into_template_p269.py --template PATH/TO/template.tex --out reports/vlm/submission_template_inserted.tex --apply",
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P269 Template Insertion Dry Run",
        "",
        f"- Template: `{result['template'] or 'not provided'}`",
        f"- Template exists: `{str(template_exists).lower()}`",
        f"- Content: `{result['content']}`",
        f"- Mode: `{mode}`",
        f"- Wrote output: `{str(wrote_output).lower()}`",
        f"- Claim check pass: `{str(check['pass']).lower()}`",
        f"- `pdflatex`: `{pdflatex or 'not found'}`",
        f"- `latexmk`: `{latexmk or 'not found'}`",
        "",
        "## Required Metrics",
    ]
    for metric, present in check["required_metrics"].items():
        lines.append(f"- `{metric}`: `{'present' if present else 'missing'}`")
    lines.extend(
        [
            "",
            "## Usage When Template Arrives",
            "- `python scripts/vlm/insert_p266_into_template_p269.py --template PATH/TO/template.tex --out reports/vlm/submission_template_inserted.tex --apply`",
            "- Then run `python scripts/vlm/check_p268_template_or_compile_readiness.py --source reports/vlm/submission_template_inserted.tex --compile` if a LaTeX engine is available.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_REPORT.relative_to(ROOT))], "mode": mode, "claim_pass": check["pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

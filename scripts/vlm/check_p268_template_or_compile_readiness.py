#!/usr/bin/env python3
"""Check whether external template/compile resources are available for P268."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "reports/vlm/p266_generic_submission_manuscript.tex"
P267_BUNDLE = ROOT / "reports/vlm/p267_submission_handoff_bundle.json"
OUT_JSON = ROOT / "reports/vlm/p268_template_or_compile_readiness.json"
OUT_MD = ROOT / "reports/vlm/p268_template_or_compile_readiness.md"

REQUIRED_METRICS = ["0.951696", "0.981566", "0.920938", "0.000000", "0.729861", "0.729524"]
FORBIDDEN_PATTERNS = [
    r"raster\s*(?:symbol\s*)?(?:detection|recognition).{0,40}(?:solved|state[- ]of[- ]the[- ]art|sota)",
    r"raster[- ]only.{0,40}0\.951696",
    r"pure\s+raster.{0,40}0\.951696",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def find_template_files(root: Path) -> list[str]:
    patterns = ["*.cls", "*.bst", "*.bbx", "*.cbx", "*.typ", "*.docx"]
    found: list[str] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            if ".git" in path.parts or ".venv" in path.parts or ".venv-vlm" in path.parts:
                continue
            found.append(str(path.relative_to(root)))
    return sorted(found)


def static_claim_check(source: Path) -> dict[str, Any]:
    text = read(source)
    lowered = text.lower()
    forbidden_hits = []
    for pattern in FORBIDDEN_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE | re.DOTALL):
            forbidden_hits.append(
                {
                    "pattern": pattern,
                    "start": match.start(),
                    "sample": text[match.start() : match.start() + 160].replace("\n", " "),
                }
            )
    return {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "exists": source.exists(),
        "required_metrics": {metric: metric in text for metric in REQUIRED_METRICS},
        "forbidden_hits": forbidden_hits,
        "has_main_boundary": "SVG/contract" in text and ("secondary" in lowered and "raster" in lowered),
        "pass": all(metric in text for metric in REQUIRED_METRICS) and not forbidden_hits,
    }


def relative_or_absolute(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def copy_compile_assets(source: Path, work_dir: Path) -> list[str]:
    copied: list[str] = []
    asset_suffixes = {".bbx", ".bib", ".bst", ".cbx", ".cls", ".eps", ".jpg", ".jpeg", ".pdf", ".png", ".sty"}
    for item in source.parent.iterdir():
        if not item.is_file() or item == source or item.suffix.lower() not in asset_suffixes:
            continue
        target = work_dir / item.name
        shutil.copy2(item, target)
        copied.append(item.name)
    return copied


def collect_compile_artifacts(work_dir: Path, out_dir: Path, stem: str) -> list[str]:
    copied: list[str] = []
    artifact_suffixes = [".pdf", ".log", ".aux", ".out", ".fls", ".fdb_latexmk", ".synctex.gz"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in artifact_suffixes:
        source_artifact = work_dir / f"{stem}{suffix}"
        if not source_artifact.exists():
            continue
        target = out_dir / source_artifact.name
        shutil.copy2(source_artifact, target)
        copied.append(relative_or_absolute(target))
    return copied


def maybe_compile(source: Path, out_dir: Path, do_compile: bool) -> dict[str, Any]:
    pdflatex = shutil.which("pdflatex")
    latexmk = shutil.which("latexmk")
    tectonic = shutil.which("tectonic")
    result: dict[str, Any] = {
        "pdflatex": pdflatex,
        "latexmk": latexmk,
        "tectonic": tectonic,
        "attempted": False,
        "returncode": None,
        "log": "",
        "work_dir": None,
        "artifacts": [],
    }
    if not do_compile:
        result["status"] = "not_requested"
        return result
    if not pdflatex and not latexmk and not tectonic:
        result["status"] = "not_attempted_no_latex_engine"
        return result
    with tempfile.TemporaryDirectory(prefix="cadbut_p268_compile_") as work_dir_name:
        work_dir = Path(work_dir_name)
        temp_source = work_dir / source.name
        shutil.copy2(source, temp_source)
        copied_assets = copy_compile_assets(source, work_dir)
        if latexmk:
            cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", temp_source.name]
        elif tectonic:
            cmd = [tectonic, "--keep-logs", "--keep-intermediates", "--outdir", str(work_dir), temp_source.name]
        else:
            cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", temp_source.name]
        try:
            proc = subprocess.run(cmd, cwd=work_dir, text=True, capture_output=True, timeout=180)
            returncode = proc.returncode
            log = proc.stdout + "\n" + proc.stderr
            status = "passed" if returncode == 0 and (work_dir / f"{temp_source.stem}.pdf").exists() else "failed"
        except subprocess.TimeoutExpired as exc:
            returncode = None
            log = (exc.stdout or "") + "\n" + (exc.stderr or "")
            status = "timeout"
        artifacts = collect_compile_artifacts(work_dir, out_dir, temp_source.stem)
    result.update(
        {
            "attempted": True,
            "returncode": returncode,
            "status": status,
            "log": log[-8000:],
            "work_dir": "temporary_local_directory",
            "copied_assets": copied_assets,
            "artifacts": artifacts,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="LaTeX source to check/compile.")
    parser.add_argument("--compile", action="store_true", help="Attempt LaTeX compile if an engine is available.")
    parser.add_argument("--out-dir", default=str(ROOT / "reports/vlm/p268_compile_out"))
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    templates = find_template_files(ROOT)
    bundle = json.loads(read(P267_BUNDLE)) if P267_BUNDLE.exists() else {}
    claim = static_claim_check(source)
    compile_result = maybe_compile(source, Path(args.out_dir), args.compile)
    ready_for_template_insertion = bool(templates)
    ready_for_compile = bool(compile_result["pdflatex"] or compile_result["latexmk"] or compile_result["tectonic"])
    result = {
        "id": "p268_template_or_compile_readiness",
        "source": relative_or_absolute(source),
        "templates_found": templates,
        "ready_for_template_insertion": ready_for_template_insertion,
        "ready_for_compile": ready_for_compile,
        "claim_check": claim,
        "compile": compile_result,
        "p267_bundle_status": bundle.get("status"),
        "decision": (
            "ready_to_compile_or_insert"
            if ready_for_template_insertion or ready_for_compile
            else "still_waiting_for_external_template_or_latex_engine"
        ),
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P268 Template Or Compile Readiness",
        "",
        f"- Source checked: `{result['source']}`",
        f"- Decision: `{result['decision']}`",
        f"- Ready for template insertion: `{str(ready_for_template_insertion).lower()}`",
        f"- Ready for compile: `{str(ready_for_compile).lower()}`",
        f"- Claim check pass: `{str(claim['pass']).lower()}`",
        "",
        "## Compile Tools",
        f"- `pdflatex`: `{compile_result['pdflatex'] or 'not found'}`",
        f"- `latexmk`: `{compile_result['latexmk'] or 'not found'}`",
        f"- `tectonic`: `{compile_result['tectonic'] or 'not found'}`",
        f"- Compile status: `{compile_result.get('status')}`",
        f"- Compile artifacts: `{', '.join(compile_result.get('artifacts', [])) or 'none'}`",
        "",
        "## Templates Found",
    ]
    if templates:
        for item in templates:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Required Metrics"])
    for metric, present in claim["required_metrics"].items():
        lines.append(f"- `{metric}`: `{'present' if present else 'missing'}`")
    lines.extend(["", "## Forbidden Claim Hits"])
    if claim["forbidden_hits"]:
        for hit in claim["forbidden_hits"]:
            lines.append(f"- `{hit['pattern']}` at {hit['start']}: {hit['sample']}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Next Step",
            "- Provide a target journal template/source or install/use a LaTeX engine, then rerun this script with `--compile` if needed."
            if result["decision"] == "still_waiting_for_external_template_or_latex_engine"
            else "- Proceed with template insertion or compile validation, then rerun claim checks.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT))], "decision": result["decision"], "claim_pass": claim["pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

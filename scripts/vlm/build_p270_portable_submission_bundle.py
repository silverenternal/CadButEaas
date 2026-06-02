#!/usr/bin/env python3
"""Build a portable archive containing the current submission-ready manuscript package."""
from __future__ import annotations

import hashlib
import json
import tarfile
import gzip
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports/vlm/submission_bundle_p270"
OUT_TAR = ROOT / "reports/vlm/p270_submission_bundle.tar.gz"
OUT_MANIFEST = ROOT / "reports/vlm/p270_submission_bundle_manifest.json"
OUT_MD = ROOT / "reports/vlm/p270_submission_bundle_manifest.md"
DETERMINISTIC_MTIME = 0

FILES = [
    "todo.json",
    "reports/vlm/p266_generic_submission_manuscript.tex",
    "reports/vlm/p266_generic_submission_static_check.md",
    "reports/vlm/p266_generic_submission_package.json",
    "reports/vlm/p275_generic_submission_compile.pdf",
    "reports/vlm/p275_generic_submission_compile.log",
    "reports/vlm/p275_generic_submission_compile.json",
    "reports/vlm/p275_generic_submission_compile.md",
    "reports/vlm/p267_submission_handoff_bundle.md",
    "reports/vlm/p267_submission_readiness_checklist.md",
    "reports/vlm/CODEX_HANDOFF_P267_SUBMISSION_READY.md",
    "reports/vlm/P272_START_HERE_EXTERNAL_RESOURCES.md",
    "reports/vlm/p272_final_external_blocker_status.md",
    "reports/vlm/p271_external_resource_request.md",
    "reports/vlm/p268_template_or_compile_readiness.md",
    "reports/vlm/p268_template_or_compile_readiness.json",
    "reports/vlm/p269_template_insertion_dryrun.md",
    "reports/vlm/p269_template_insertion_dryrun.json",
    "reports/vlm/p265_final_manuscript_package.md",
    "reports/vlm/p265_static_claim_consistency_check.md",
    "reports/vlm/p264_final_results_tables.md",
    "reports/vlm/p264_final_claim_integration_snippets.md",
    "reports/vlm/p263_secondary_raster_adapter_package.md",
    "reports/vlm/p247a_svg_contract_metric_package.md",
    "reports/vlm/p247b_svg_contract_ablation_pack.md",
    "reports/vlm/p247c_svg_weak_label_audit.md",
    "scripts/vlm/check_p268_template_or_compile_readiness.py",
    "scripts/vlm/insert_p266_into_template_p269.py",
    "scripts/vlm/build_p267_submission_handoff_bundle.py",
    "scripts/vlm/build_p270_portable_submission_bundle.py",
    "scripts/vlm/resume_submission_p271.sh",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = DETERMINISTIC_MTIME
    if info.isdir():
        info.mode = 0o755
    else:
        info.mode = 0o644
    return info


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    missing = []
    for rel in FILES:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = OUT_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        entries.append(
            {
                "path": rel,
                "bytes": src.stat().st_size,
                "sha256": sha256(src),
            }
        )

    readme = OUT_DIR / "README_SUBMISSION_BUNDLE.md"
    readme.write_text(
        "\n".join(
            [
                "# CadStruct-MoE Submission Bundle P270",
                "",
                "This bundle contains the current project-side submission package.",
                "",
                "## Start Here",
                "- `reports/vlm/p266_generic_submission_manuscript.tex` is the generic LaTeX manuscript.",
                "- `reports/vlm/P272_START_HERE_EXTERNAL_RESOURCES.md` is the first handoff document to read.",
                "- `reports/vlm/p267_submission_handoff_bundle.md` lists the core files and blockers.",
                "- `reports/vlm/p268_template_or_compile_readiness.md` records current template/compiler readiness.",
                "- `scripts/vlm/resume_submission_p271.sh` reruns readiness checks, template insertion, and bundle refresh.",
                "- `scripts/vlm/insert_p266_into_template_p269.py` can insert the generic manuscript into a future LaTeX template.",
                "",
                "## Current Blockers",
                "- No venue-specific journal template/source was present in the repo when this bundle was built.",
                "- The generic manuscript has a successful P275 compile artifact; remaining compile work is template-specific.",
                "",
                "## Guardrails",
                "- Main claim: SVG/contract CadStruct-MoE.",
                "- Secondary claim: bounded P262 runtime raster adapter bridge.",
                "- Do not report P259 diagnostic upper bounds as official metrics.",
                "- Do not claim raster detector SOTA.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    entries.append(
        {
            "path": "README_SUBMISSION_BUNDLE.md",
            "bytes": readme.stat().st_size,
            "sha256": sha256(readme),
        }
    )

    if OUT_TAR.exists():
        OUT_TAR.unlink()
    with OUT_TAR.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=DETERMINISTIC_MTIME) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in sorted(OUT_DIR.rglob("*"), key=lambda item: item.relative_to(OUT_DIR).as_posix()):
                    arcname = Path("submission_bundle_p270") / path.relative_to(OUT_DIR)
                    tar.add(path, arcname=arcname.as_posix(), filter=reset_tarinfo)

    result = {
        "id": "p270_portable_submission_bundle",
        "bundle_dir": str(OUT_DIR.relative_to(ROOT)),
        "tarball": str(OUT_TAR.relative_to(ROOT)),
        "tarball_bytes": OUT_TAR.stat().st_size,
        "tarball_sha256": sha256(OUT_TAR),
        "file_count": len(entries),
        "missing": missing,
        "entries": entries,
        "claim_boundary": "Portable bundle for template insertion/compile once external resources are available.",
    }
    OUT_MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P270 Portable Submission Bundle Manifest",
        "",
        f"- Bundle dir: `{result['bundle_dir']}`",
        f"- Tarball: `{result['tarball']}`",
        f"- Tarball bytes: `{result['tarball_bytes']}`",
        f"- Tarball SHA256: `{result['tarball_sha256']}`",
        f"- File count: `{result['file_count']}`",
        f"- Missing files: `{len(missing)}`",
        "",
        "## Included Files",
        "| file | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for entry in entries:
        lines.append(f"| `{entry['path']}` | {entry['bytes']} | `{entry['sha256']}` |")
    if missing:
        lines.extend(["", "## Missing"])
        for item in missing:
            lines.append(f"- `{item}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_MANIFEST.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT)), str(OUT_TAR.relative_to(ROOT))], "missing": len(missing), "tarball_bytes": result["tarball_bytes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

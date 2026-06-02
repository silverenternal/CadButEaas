#!/usr/bin/env python3
"""Audit external floor-plan datasets staged for the raster MoE rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOTS = {
    "cubicasa5k": Path("datasets/external/cubicasa5k"),
    "cubicasa5k_legacy_zenodo": Path("datasets/external/cubicasa5k_zenodo"),
    "cubicasa5k_legacy_hf": Path("datasets/external/cubicasa5k_hf"),
    "bridge": Path("datasets/external/bridge/repo"),
    "floorplancad": Path("datasets/external/floorplancad"),
    "mlstructfp": Path("datasets/external/mlstructfp"),
    "cvc_fp": Path("datasets/external/cvc_fp"),
    "cvc_fp_legacy_figshare": Path("datasets/external/cvc_fp_figshare"),
    "resplan": Path("datasets/external/resplan"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/vlm/external_floorplan_datasets_v18_audit.json")
    args = parser.parse_args()

    audit = {
        "schema": "external_floorplan_datasets_v18_audit",
        "datasets": {
            "cubicasa5k": audit_cubicasa(ROOTS["cubicasa5k"]),
            "cubicasa5k_legacy_zenodo": audit_cubicasa_legacy_zenodo(ROOTS["cubicasa5k_legacy_zenodo"]),
            "cubicasa5k_legacy_hf": audit_cubicasa_legacy_hf(ROOTS["cubicasa5k_legacy_hf"]),
            "bridge": audit_bridge(ROOTS["bridge"]),
            "floorplancad": audit_floorplancad(ROOTS["floorplancad"]),
            "mlstructfp": audit_mlstructfp(ROOTS["mlstructfp"]),
            "cvc_fp": audit_cvc_fp(ROOTS["cvc_fp"]),
            "cvc_fp_legacy_figshare": audit_cvc_fp(ROOTS["cvc_fp_legacy_figshare"]),
            "resplan": audit_resplan(ROOTS["resplan"]),
        },
        "converted_datasets": audit_converted_datasets(),
        "usage_priority": [
            {
                "dataset": "cubicasa5k",
                "priority": "P0",
                "role": "derive room polygon, symbol/icon instance, and contains_symbol assignment supervision from raster images plus offline SVG labels",
            },
            {
                "dataset": "bridge",
                "priority": "P0_aux",
                "role": "symbol spotting, room classification, and region caption auxiliary pretraining; not a primary contains_symbol graph dataset",
            },
            {
                "dataset": "floorplancad",
                "priority": "P0_aux",
                "role": "large symbol/type pretraining from rasterized CAD pages with offline detection labels",
            },
            {
                "dataset": "mlstructfp",
                "priority": "P1_blocked",
                "role": "wall/slab structural pretraining after the public download link is requested through the dataset form",
            },
            {
                "dataset": "cvc_fp",
                "priority": "P1",
                "role": "small structural floor-plan sanity/evaluation set if the original package can be retrieved",
            },
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def audit_cubicasa(root: Path) -> dict[str, Any]:
    repo = root / "repo"
    zip_path = root / "raw" / "cubicasa5k.zip"
    data = {
        "root": str(root),
        "repo_present": (repo / ".git").exists(),
        "repo_head": git_head(repo),
        "zip_path": str(zip_path),
        "zip_present": zip_path.exists(),
        "zip_size_bytes": file_size(zip_path),
        "expected_zip_size_bytes": 5_469_495_706,
        "expected_md5": "0ce0b203d1e3c125b51087b219bd23b9",
        "license": "CC-BY-NC-SA-4.0",
        "download_url": "https://zenodo.org/api/records/2613548/files/cubicasa5k.zip/content",
        "status": "downloaded" if zip_path.exists() and file_size(zip_path) == 5_469_495_706 else "download_incomplete_or_missing",
    }
    if data["zip_present"] and data["zip_size_bytes"] == data["expected_zip_size_bytes"]:
        data["md5"] = md5(zip_path)
        data["md5_ok"] = data["md5"] == data["expected_md5"]
    return data


def audit_cubicasa_legacy_zenodo(root: Path) -> dict[str, Any]:
    zip_path = root / "cubicasa5k.zip"
    unpacked = root / "unpacked" / "cubicasa5k"
    data = {
        "root": str(root),
        "zip_path": str(zip_path),
        "zip_present": zip_path.exists(),
        "zip_size_bytes": file_size(zip_path),
        "expected_zip_size_bytes": 5_469_495_706,
        "expected_md5": "0ce0b203d1e3c125b51087b219bd23b9",
        "unpacked_present": unpacked.exists(),
        "split_files_present": {
            "train.txt": (unpacked / "train.txt").exists(),
            "val.txt": (unpacked / "val.txt").exists(),
            "test.txt": (unpacked / "test.txt").exists(),
        },
        "status": "available_legacy_download" if zip_path.exists() and unpacked.exists() else "missing_or_partial",
    }
    if data["zip_present"] and data["zip_size_bytes"] == data["expected_zip_size_bytes"]:
        data["md5"] = md5(zip_path)
        data["md5_ok"] = data["md5"] == data["expected_md5"]
    return data


def audit_cubicasa_legacy_hf(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "present": root.exists(),
        "image_files": count_suffixes(root, {".png", ".jpg", ".jpeg"}),
        "json_files": count_suffixes(root, {".json", ".jsonl"}),
        "status": "available_legacy_hf" if root.exists() else "missing",
    }


def audit_bridge(root: Path) -> dict[str, Any]:
    base = root / "dataset_bridge"
    return {
        "root": str(root),
        "repo_present": (root / ".git").exists(),
        "repo_head": git_head(root),
        "image_files": count_suffixes(base, {".jpg", ".jpeg", ".png"}),
        "xml_files": count_suffixes(base, {".xml"}),
        "json_files": count_suffixes(base, {".json"}),
        "top_level_tasks": sorted(path.name for path in base.iterdir() if path.is_dir()) if base.exists() else [],
        "status": "downloaded_repo_data" if base.exists() else "missing",
        "role_note": "Use symbol spotting and room/region auxiliary labels; do not treat it as a full topology graph dataset.",
    }


def audit_floorplancad(root: Path) -> dict[str, Any]:
    samples_path = root / "samples.json"
    samples = 0
    if samples_path.exists():
        try:
            samples = len(json.loads(samples_path.read_text(encoding="utf-8")).get("samples", []))
        except Exception:
            samples = 0
    return {
        "root": str(root),
        "samples_json_present": samples_path.exists(),
        "sample_count": samples,
        "image_files": count_suffixes(root / "data", {".png", ".jpg", ".jpeg"}),
        "status": "available" if samples else "missing_or_unreadable",
    }


def audit_mlstructfp(root: Path) -> dict[str, Any]:
    repo = root / "repo"
    return {
        "root": str(root),
        "repo_present": (repo / ".git").exists(),
        "repo_head": git_head(repo),
        "data_zip_present": any(root.glob("*.zip")),
        "license": "MIT for loader code; dataset download requires public link request form",
        "download_blocker": "Dataset zip is not directly downloadable from the repo; README requires requesting a public link through https://forms.gle/HigdGxngnTEvnNC37.",
        "status": "repo_downloaded_data_blocked",
    }


def audit_cvc_fp(root: Path) -> dict[str, Any]:
    return {
        "root": str(root),
        "image_files": count_suffixes(root, {".png", ".jpg", ".jpeg", ".tif", ".tiff"}),
        "shape_files": count_suffixes(root, {".shp"}),
        "zip_files": count_suffixes(root, {".zip"}),
        "status": "available" if any(root.glob("**/*.shp")) else "not_downloaded",
        "role_note": "Small structural floor-plan sanity set; keep as audit/eval, not main training.",
    }


def audit_resplan(root: Path) -> dict[str, Any]:
    zip_path = root / "ResPlan.zip"
    pkl_path = root / "ResPlan.pkl"
    return {
        "root": str(root),
        "zip_present": zip_path.exists(),
        "zip_size_bytes": file_size(zip_path),
        "pkl_present": pkl_path.exists(),
        "pkl_size_bytes": file_size(pkl_path),
        "license_present": (root / "LICENSE").exists(),
        "readme_present": (root / "README.md").exists(),
        "status": "available" if pkl_path.exists() or zip_path.exists() else "missing",
        "role_note": "Room topology/layout graph source; useful for adjacency and spatial priors, not direct symbol-room containment.",
    }


def audit_converted_datasets() -> dict[str, Any]:
    targets = {
        "cadstruct_cubicasa5k_moe": Path("datasets/cadstruct_cubicasa5k_moe"),
        "cadstruct_cubicasa5k_moe_locked": Path("datasets/cadstruct_cubicasa5k_moe_locked"),
        "cadstruct_rooms_v1": Path("datasets/cadstruct_rooms_v1"),
        "cadstruct_symbols_v1": Path("datasets/cadstruct_symbols_v1"),
        "cadstruct_graph_nodes_paper_v2_cvc_train_floor_test": Path(
            "datasets/cadstruct_graph_nodes_paper_v2_cvc_train_floor_test"
        ),
        "cadstruct_graph_nodes_paper_v2_floor_train_cvc_test": Path(
            "datasets/cadstruct_graph_nodes_paper_v2_floor_train_cvc_test"
        ),
        "floorplancad_symbol_pretrain_v16": Path("datasets/floorplancad_symbol_pretrain_v16"),
    }
    return {name: audit_converted_dataset(path) for name, path in targets.items()}


def audit_converted_dataset(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.glob("*.jsonl")) if root.exists() else []
    manifest = root / "manifest.json"
    result: dict[str, Any] = {
        "root": str(root),
        "present": root.exists(),
        "jsonl_files": [path.name for path in files],
        "jsonl_total_bytes": sum(path.stat().st_size for path in files),
        "manifest_present": manifest.exists(),
        "status": "available" if files else "missing",
    }
    if manifest.exists():
        try:
            result["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            result["manifest_error"] = str(exc)
    return result


def count_suffixes(root: Path, suffixes: set[str]) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.exists():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = root / ".git" / value.removeprefix("ref: ")
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else value
    return value


if __name__ == "__main__":
    main()

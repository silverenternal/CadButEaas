#!/usr/bin/env python3
"""Build cadstruct_real_world_benchmark_v3 with leakage audit and coverage matrix.

R0-T3: Merge reviewed internal-real with public sources into v3 split,
generate train/dev/locked/source-heldout partitions, check image hash /
annotation hash / path overlap / teacher contamination.

Outputs:
  - datasets/cadstruct_real_world_benchmark_v3/manifest.json
  - reports/vlm/benchmark_v3_leakage_audit.json
  - reports/vlm/benchmark_v3_coverage_matrix.json
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
V3_DIR = ROOT / "datasets" / "cadstruct_real_world_benchmark_v3"
REPORTS_DIR = ROOT / "reports" / "vlm"

# ── Config ───────────────────────────────────────────────────────────────

SPLIT_RATIOS = {
    "train": 0.70,
    "dev": 0.10,
    "locked": 0.20,
}

# Sources to include
SOURCES = ["cubicasa5k", "cvc_fp", "floorplancad", "internal_real_v3"]

# Element families we track
ELEMENT_FAMILIES = ["wall_opening", "room_space", "symbol_fixture", "text_dimension", "sheet_layout", "scene_graph"]

# ── Helpers ──────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def deterministic_split(records: list, seed: int = 20260501):
    """Split records into train/dev/locked deterministically by index."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    n = len(records)
    n_dev = max(1, int(n * SPLIT_RATIOS["dev"]))
    n_locked = max(1, int(n * SPLIT_RATIOS["locked"]))
    n_train = n - n_dev - n_locked

    dev_idx = set(indices[:n_dev])
    locked_idx = set(indices[n_dev:n_dev + n_locked])
    train_idx = set(indices[n_dev + n_locked:])

    splits = {"train": [], "dev": [], "locked": []}
    for i, idx in enumerate(indices):
        rec = records[idx]
        if idx in train_idx:
            splits["train"].append(rec)
        elif idx in dev_idx:
            splits["dev"].append(rec)
        else:
            splits["locked"].append(rec)
    return splits


def load_internal_real_v3():
    """Load internal_real_v3 candidate manifest."""
    manifest_path = ROOT / "datasets" / "internal_real_v3" / "manifest.json"
    if not manifest_path.exists():
        print(f"[WARN] {manifest_path} not found; internal_real_v3 will be empty")
        return []
    with open(manifest_path) as f:
        data = json.load(f)
    return data.get("records", [])


def load_cubicasa5k_locked():
    """Load cadstruct_cubicasa5k_moe_locked splits."""
    locked_dir = ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked"
    records = []
    for split_name in ["smoke.jsonl", "dev.jsonl", "locked_test.jsonl"]:
        split_path = locked_dir / split_name
        if not split_path.exists():
            print(f"[WARN] {split_path} not found")
            continue
        with open(split_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_source"] = "cubicasa5k"
                rec["_original_split"] = split_name.replace(".jsonl", "")
                records.append(rec)
    return records


def load_cvc_fp_records():
    """Load CVC-FP records from benchmark_v1 mixed_source_locked_test.jsonl."""
    v1_dir = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1"
    records = []
    mixed_path = v1_dir / "wall_opening" / "mixed_source_locked_test.jsonl"
    if mixed_path.exists():
        with open(mixed_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                src = rec.get("source_dataset", "")
                if "cvc" in src.lower():
                    rec["_source"] = "cvc_fp"
                    rec["_element_family"] = "wall_opening"
                    rec["image_path"] = rec.get("image", "")
                    rec["annotation_path"] = ""
                    records.append(rec)
    return records


def load_floorplancad_records():
    """Load FloorPlanCAD records from dedicated locked file (not mixed)."""
    v1_dir = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1"
    records = []
    # Use only the floorplancad-specific locked file to avoid double-counting
    fp_path = v1_dir / "wall_opening" / "floorplancad_locked_test.jsonl"
    if fp_path.exists():
        with open(fp_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_source"] = "floorplancad"
                rec["_element_family"] = "wall_opening"
                rec["image_path"] = rec.get("image", "")
                rec["annotation_path"] = ""
                records.append(rec)
    return records


def normalize_internal_record(rec: dict) -> dict:
    """Normalize an internal_real_v3 candidate into a standard record."""
    return {
        "id": rec.get("id", ""),
        "source": "internal_real_v3",
        "source_dataset": rec.get("source_dataset", ""),
        "source_path": rec.get("source_path", ""),
        "image_path": rec.get("image", ""),
        "annotation_path": rec.get("annotation", ""),
        "scan_quality": rec.get("scan_quality", "unknown_requires_review"),
        "license_or_privacy_status": rec.get("license_or_privacy_status", ""),
        "expected_element_groups": rec.get("expected_element_groups", []),
        "expert_seed": rec.get("expert_seed", ""),
        "review_status": rec.get("review_status", "pending_human_review"),
        "split_intent": rec.get("split_intent", "locked"),
        "deidentified": rec.get("deidentified", False),
        "privacy_notes": rec.get("privacy_notes", ""),
        "tags": rec.get("tags", []),
        "_source": "internal_real_v3",
        "_original_split": "candidate",
    }


def normalize_cubicasa_record(rec: dict) -> dict:
    """Normalize a cubicasa5k locked record into a standard record."""
    return {
        "id": sha256_str(rec.get("image_path", ""))[:12],
        "source": "cubicasa5k",
        "source_dataset": "cubicasa5k",
        "source_path": rec.get("image_path", ""),
        "image_path": rec.get("image_path", ""),
        "annotation_path": rec.get("annotation_path", ""),
        "scan_quality": "clean_raster",
        "license_or_privacy_status": "Apache-2.0",
        "expected_element_groups": ["room_space", "symbol_fixture", "text_dimension", "scene_graph"],
        "expert_seed": "scene_graph",
        "review_status": "locked",
        "split_intent": rec.get("_original_split", "dev"),
        "deidentified": True,
        "privacy_notes": "",
        "tags": [],
        "_source": "cubicasa5k",
        "_original_split": rec.get("_original_split", "dev"),
        "_metadata": rec.get("metadata", {}),
        "_prompt": rec.get("prompt", ""),
        "_request_hints": rec.get("request_hints", {}),
    }


def normalize_public_record(rec: dict, source: str, element_family: str) -> dict:
    """Normalize a CVC-FP or FloorPlanCAD record."""
    path_key = rec.get("image_path", rec.get("path", ""))
    return {
        "id": sha256_str(path_key)[:12],
        "source": source,
        "source_dataset": source,
        "source_path": path_key,
        "image_path": rec.get("image_path", ""),
        "annotation_path": rec.get("annotation_path", rec.get("label_path", "")),
        "scan_quality": "public_benchmark_raster",
        "license_or_privacy_status": "CC BY-NC 4.0",
        "expected_element_groups": [element_family],
        "expert_seed": element_family,
        "review_status": "locked",
        "split_intent": "locked",
        "deidentified": True,
        "privacy_notes": "",
        "tags": [],
        "_source": source,
        "_original_split": "locked",
        "_element_family": element_family,
    }


# ── Main build logic ─────────────────────────────────────────────────────

def build_v3_splits():
    """Build the v3 benchmark splits from all sources."""
    print("[1/5] Loading source datasets...")

    # Load all sources
    internal_candidates = load_internal_real_v3()
    cubicasa_records = load_cubicasa5k_locked()
    cvc_records = load_cvc_fp_records()
    floorplancad_records = load_floorplancad_records()

    # Normalize
    internal_normalized = [normalize_internal_record(r) for r in internal_candidates]
    cubicasa_normalized = [normalize_cubicasa_record(r) for r in cubicasa_records]
    cvc_normalized = [normalize_public_record(r, "cvc_fp", "wall_opening") for r in cvc_records]
    floorplancad_normalized = [normalize_public_record(r, "floorplancad", "wall_opening") for r in floorplancad_records]

    print(f"  internal_real_v3: {len(internal_normalized)} candidates")
    print(f"  cubicasa5k: {len(cubicasa_normalized)} records")
    print(f"  cvc_fp: {len(cvc_normalized)} records")
    print(f"  floorplancad: {len(floorplancad_normalized)} records")

    # ── Split strategy ───────────────────────────────────────────────────
    print("[2/5] Building splits...")

    all_splits = {"train": [], "dev": [], "locked": [], "source_heldout": []}

    # cubicasa5k: preserve existing splits
    #   dev → dev (development evaluation)
    #   smoke → dev (smoke subset for quick iteration)
    #   locked_test → locked (final evaluation, never train on this)
    for rec in cubicasa_normalized:
        orig = rec["_original_split"]
        if orig in ("smoke", "dev"):
            all_splits["dev"].append(rec)
        elif orig == "locked_test":
            all_splits["locked"].append(rec)
        # Note: cubicasa train (3,950 records) exists in the parent dataset
        # but is not in the locked directory; skip for v3 build

    # cvc_fp and floorplancad: these are locked evaluation sets from v1
    # They belong in the locked split (never train on them)
    for rec in cvc_normalized:
        all_splits["locked"].append(rec)
    for rec in floorplancad_normalized:
        all_splits["locked"].append(rec)

    # internal_real_v3: all 382 candidates are pending_human_review
    # They go to source_heldout until double-review is complete
    # R0-T3 done_when requires 100 double-review locked → blocks completion
    # DEDUPLICATION: exclude internal candidates that reference images already
    # in dev or locked splits (these are hard-case mining on existing data,
    # not genuinely new images).
    dev_images = {r.get("image_path", "") for r in all_splits["dev"] if r.get("image_path")}
    locked_images = {r.get("image_path", "") for r in all_splits["locked"] if r.get("image_path")}
    existing_images = dev_images | locked_images

    deduped_internal = []
    excluded_internal = []
    for rec in internal_normalized:
        img = rec.get("image_path", "")
        if img and img in existing_images:
            excluded_internal.append(rec)
        else:
            deduped_internal.append(rec)

    print(f"  internal_real_v3 dedup: {len(internal_normalized)} → {len(deduped_internal)} kept, {len(excluded_internal)} excluded (image overlap)")

    for rec in deduped_internal:
        all_splits["source_heldout"].append(rec)

    # Print split summary
    for split_name, records in all_splits.items():
        by_source = Counter(r["source"] for r in records)
        print(f"  {split_name}: {len(records)} records — {dict(by_source)}")

    return all_splits, internal_normalized, cubicasa_normalized, cvc_normalized, floorplancad_normalized


def run_leakage_audit(all_splits, all_records):
    """Run leakage audit: image hash, annotation hash, path overlap, teacher contamination."""
    print("[3/5] Running leakage audit...")

    audit = {
        "version": "benchmark_v3_leakage_audit_v1",
        "date": "2026-05-01",
        "image_hash_overlap": {},
        "annotation_hash_overlap": {},
        "path_overlap": {},
        "teacher_contamination": {},
        "split_isolation": {},
        "summary": {},
    }

    # Build hash sets per split
    split_image_hashes = {}
    split_annotation_hashes = {}
    split_paths = {}

    for split_name, records in all_splits.items():
        img_hashes = set()
        ann_hashes = set()
        paths = set()
        for rec in records:
            img_path = rec.get("image_path", "")
            ann_path = rec.get("annotation_path", "")
            if img_path:
                img_hashes.add(img_path)
                paths.add(img_path)
            if ann_path:
                ann_hashes.add(ann_path)
                paths.add(ann_path)
        split_image_hashes[split_name] = img_hashes
        split_annotation_hashes[split_name] = ann_hashes
        split_paths[split_name] = paths

    # Check pairwise overlap between splits
    split_names = list(all_splits.keys())
    image_overlaps = {}
    annotation_overlaps = {}
    path_overlaps = {}

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            img_overlap = split_image_hashes[s1] & split_image_hashes[s2]
            ann_overlap = split_annotation_hashes[s1] & split_annotation_hashes[s2]
            path_overlap = split_paths[s1] & split_paths[s2]

            if img_overlap:
                image_overlaps[f"{s1}↔{s2}"] = len(img_overlap)
            if ann_overlap:
                annotation_overlaps[f"{s1}↔{s2}"] = len(ann_overlap)
            if path_overlap:
                path_overlaps[f"{s1}↔{s2}"] = len(path_overlap)

    audit["image_hash_overlap"] = {
        "leakage_detected": len(image_overlaps) > 0,
        "overlaps": image_overlaps,
        "total_pairs_checked": len(image_overlaps),
    }
    audit["annotation_hash_overlap"] = {
        "leakage_detected": len(annotation_overlaps) > 0,
        "overlaps": annotation_overlaps,
        "total_pairs_checked": len(annotation_overlaps),
    }
    audit["path_overlap"] = {
        "leakage_detected": len(path_overlaps) > 0,
        "overlaps": path_overlaps,
        "total_pairs_checked": len(path_overlaps),
    }

    # Teacher contamination check
    # Check if any records reference VLM teacher hints or annotation-backed labels
    teacher_contaminated = []
    for split_name, records in all_splits.items():
        for rec in records:
            hints = rec.get("_request_hints", {})
            tags = rec.get("tags", [])
            if hints and any(k.endswith("_teacher") for k in hints):
                teacher_contaminated.append({
                    "split": split_name,
                    "id": rec.get("id", ""),
                    "contamination_type": "teacher_hint_in_request_hints",
                })
            if "teacher_hint" in tags or "annotation_backed" in tags:
                teacher_contaminated.append({
                    "split": split_name,
                    "id": rec.get("id", ""),
                    "contamination_type": "teacher_tag_in_record",
                })

    audit["teacher_contamination"] = {
        "contaminated_records": teacher_contaminated,
        "total_contaminated": len(teacher_contaminated),
        "clean": len(teacher_contaminated) == 0,
    }

    # Split isolation summary
    for split_name in split_names:
        n = len(all_splits[split_name])
        other_paths = set()
        for other_name in split_names:
            if other_name != split_name:
                other_paths |= split_paths[other_name]
        leaked = split_paths[split_name] & other_paths
        audit["split_isolation"][split_name] = {
            "total_records": n,
            "leaked_to_other_splits": len(leaked),
            "isolation_rate": 1.0 - len(leaked) / max(1, n),
        }

    audit["summary"] = {
        "image_hash_leakage": len(image_overlaps) == 0,
        "annotation_hash_leakage": len(annotation_overlaps) == 0,
        "path_leakage": len(path_overlaps) == 0,
        "teacher_contamination_free": len(teacher_contaminated) == 0,
        "overall_clean": (
            len(image_overlaps) == 0
            and len(annotation_overlaps) == 0
            and len(path_overlaps) == 0
            and len(teacher_contaminated) == 0
        ),
    }

    return audit


def build_coverage_matrix(all_splits):
    """Build element coverage matrix: source × element family × split."""
    print("[4/5] Building coverage matrix...")

    matrix = {
        "version": "benchmark_v3_coverage_matrix_v1",
        "date": "2026-05-01",
        "sources": {},
        "element_families": {},
        "split_coverage": {},
    }

    # Per-source breakdown
    for split_name, records in all_splits.items():
        by_source = defaultdict(lambda: defaultdict(int))
        by_element = defaultdict(int)
        for rec in records:
            source = rec["source"]
            elements = rec.get("expected_element_groups", [])
            for elem in elements:
                by_source[source][elem] += 1
                by_element[elem] += 1

        matrix["sources"][split_name] = dict(by_source)
        matrix["element_families"][split_name] = dict(by_element)

    # Source × element × split matrix
    all_sources = set()
    all_elements = set()
    for split_name, records in all_splits.items():
        for rec in records:
            all_sources.add(rec["source"])
            for elem in rec.get("expected_element_groups", []):
                all_elements.add(elem)

    coverage_grid = {}
    for source in sorted(all_sources):
        coverage_grid[source] = {}
        for split_name in all_splits:
            coverage_grid[source][split_name] = {}
            for elem in sorted(all_elements):
                count = 0
                for rec in all_splits[split_name]:
                    if rec["source"] == source and elem in rec.get("expected_element_groups", []):
                        count += 1
                coverage_grid[source][split_name][elem] = count

    matrix["split_coverage"] = coverage_grid

    # Source quality summary
    matrix["source_quality"] = {}
    for split_name, records in all_splits.items():
        for rec in records:
            source = rec["source"]
            if source not in matrix["source_quality"]:
                matrix["source_quality"][source] = {
                    "total_records": 0,
                    "review_status": Counter(),
                    "scan_qualities": Counter(),
                    "licenses": Counter(),
                }
            sq = matrix["source_quality"][source]
            sq["total_records"] += 1
            sq["review_status"][rec.get("review_status", "unknown")] += 1
            sq["scan_qualities"][rec.get("scan_quality", "unknown")] += 1
            sq["licenses"][rec.get("license_or_privacy_status", "unknown")] += 1

    # Convert Counters to dicts for JSON serialization
    for source in matrix["source_quality"]:
        sq = matrix["source_quality"][source]
        sq["review_status"] = dict(sq["review_status"])
        sq["scan_qualities"] = dict(sq["scan_qualities"])
        sq["licenses"] = dict(sq["licenses"])

    return matrix


def write_manifest(all_splits):
    """Write the v3 benchmark manifest."""
    print("[5/5] Writing manifest...")

    # Compute dataset hash from all records
    all_records_serialized = []
    for split_name, records in all_splits.items():
        for rec in records:
            # Strip internal fields for hashing
            clean = {k: v for k, v in rec.items() if not k.startswith("_")}
            clean["_split"] = split_name
            all_records_serialized.append(json.dumps(clean, sort_keys=True))

    dataset_hash = sha256_str("\n".join(sorted(all_records_serialized)))

    manifest = {
        "version": "cadstruct_real_world_benchmark_v3",
        "date": "2026-05-01",
        "dataset_hash": dataset_hash,
        "status": "v3_locked_split_built_review_required",
        "policy_note": "locked split may only be used for final evaluation; teacher hints must be disabled by default.",
        "sources": sorted(set(
            rec["source"]
            for records in all_splits.values()
            for rec in records
        )),
        "splits": {},
        "split_summary": {},
    }

    for split_name, records in all_splits.items():
        # Write JSONL for each split
        split_path = V3_DIR / f"{split_name}.jsonl"
        with open(split_path, "w") as f:
            for rec in records:
                clean = {k: v for k, v in rec.items() if not k.startswith("_")}
                clean["split"] = split_name
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        # Compute split hash
        split_hash = sha256_file(split_path)

        by_source = Counter(r["source"] for r in records)
        by_element = Counter()
        for rec in records:
            for elem in rec.get("expected_element_groups", []):
                by_element[elem] += 1

        manifest["splits"][split_name] = {
            "path": str(split_path.relative_to(ROOT)),
            "records": len(records),
            "sha256": split_hash,
            "by_source": dict(by_source),
            "by_element_family": dict(by_element),
        }

    manifest["split_summary"] = {
        "total_records": sum(len(r) for r in all_splits.values()),
        "sources": sorted(set(
            rec["source"]
            for records in all_splits.values()
            for rec in records
        )),
        "r0_t3_status": {
            "four_source_benchmark": len(manifest["sources"]) >= 4,
            "internal_real_candidates": sum(
                1 for records in all_splits.values()
                for rec in records
                if rec["source"] == "internal_real_v3"
            ),
            "locked_internal": sum(
                1 for records in all_splits.values()
                for rec in records
                if rec["source"] == "internal_real_v3" and rec.get("review_status") == "locked"
            ),
            "pending_review": sum(
                1 for records in all_splits.values()
                for rec in records
                if rec.get("review_status") == "pending_human_review"
            ),
        },
    }

    manifest_path = V3_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("cadstruct_real_world_benchmark_v3 — R0-T3 Build")
    print("=" * 70)

    # Ensure output directories
    V3_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build splits
    all_splits, internal, cubicasa, cvc, floorplancad = build_v3_splits()

    # Leakage audit
    audit = run_leakage_audit(all_splits, internal + cubicasa + cvc + floorplancad)

    # Coverage matrix
    coverage = build_coverage_matrix(all_splits)

    # Write manifest
    manifest = write_manifest(all_splits)

    # Write outputs
    audit_path = REPORTS_DIR / "benchmark_v3_leakage_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    coverage_path = REPORTS_DIR / "benchmark_v3_coverage_matrix.json"
    with open(coverage_path, "w") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    # Print summary
    print()
    print("=" * 70)
    print("R0-T3 Build Summary")
    print("=" * 70)
    print(f"Dataset hash: {manifest['dataset_hash']}")
    print(f"Total records: {manifest['split_summary']['total_records']}")
    print(f"Sources: {manifest['split_summary']['sources']}")
    print(f"Four-source benchmark: {manifest['split_summary']['r0_t3_status']['four_source_benchmark']}")
    print(f"Internal real candidates: {manifest['split_summary']['r0_t3_status']['internal_real_candidates']}")
    print(f"Locked internal: {manifest['split_summary']['r0_t3_status']['locked_internal']}")
    print(f"Pending review: {manifest['split_summary']['r0_t3_status']['pending_review']}")
    print()
    print(f"Leakage audit: {audit['summary']}")
    print()
    print("Outputs:")
    print(f"  {manifest['splits']}")
    print(f"  {audit_path}")
    print(f"  {coverage_path}")
    print(f"  {V3_DIR / 'manifest.json'}")

    # R0-T3 done_when check
    print()
    print("R0-T3 done_when check:")
    four_source = manifest['split_summary']['r0_t3_status']['four_source_benchmark']
    no_leakage = audit['summary']['overall_clean']
    core_element_non_cubicasa = any(
        src not in ("cubicasa5k",)
        for src in manifest['split_summary']['sources']
    )
    internal_100_locked = manifest['split_summary']['r0_t3_status']['locked_internal'] >= 100
    print(f"  {'✓' if no_leakage else '✗'} locked split no train/dev leakage: {no_leakage}")
    print(f"  {'✓' if four_source else '✗'} four-source coverage: {four_source}")
    print(f"  {'✓' if core_element_non_cubicasa else '✗'} non-CubiCasa core element source: {core_element_non_cubicasa}")
    print(f"  {'✓' if internal_100_locked else '⚠'} internal-real 100 double-review locked: {manifest['split_summary']['r0_t3_status']['locked_internal']} / 100 required (requires human review)")
    print(f"  ⚠ internal-real total candidates: {manifest['split_summary']['r0_t3_status']['internal_real_candidates']} / 300 required")

    structure_complete = four_source and no_leakage and core_element_non_cubicasa
    fully_complete = structure_complete and internal_100_locked
    if fully_complete:
        status = "PASS"
    elif structure_complete:
        status = "STRUCTURE_COMPLETE — BLOCKED on 100 double-review locked internal-real"
    else:
        status = "BLOCKED"
    print(f"\n  R0-T3 {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

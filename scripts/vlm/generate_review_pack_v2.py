#!/usr/bin/env python3
"""
R0-T3: Generate double-review locked record pack for human annotation.

Reads benchmark_v3 internal_real_v3 candidates, filters for double-review
eligibility, and produces a review queue with uncertainty scoring, HTML
preview, and CSV export suitable for human reviewers.

Usage:
    python scripts/vlm/generate_review_pack_v2.py

Outputs:
    reports/vlm/internal_real_v3_review_pack_v2/review_queue.jsonl
    reports/vlm/internal_real_v3_review_pack_v2/review.html
    reports/vlm/internal_real_v3_review_pack_v2/review_queue.csv
    reports/vlm/internal_real_v3_review_pack_v2/review_pack_audit.json
"""

import json
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_MANIFEST = ROOT / "datasets" / "cadstruct_real_world_benchmark_v3" / "manifest.json"
INTERNAL_REAL_MANIFEST = ROOT / "datasets" / "internal_real_v3" / "manifest.json"
REVIEW_DIR = ROOT / "reports" / "vlm" / "internal_real_v3_review_pack_v2"
TARGET_LOCKED = 100


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_review_queue():
    """Build review queue from internal_real_v3 candidates."""
    print("=" * 70)
    print("R0-T3: Generate Double-Review Locked Record Pack")
    print("=" * 70)

    # Load manifests
    if BENCHMARK_MANIFEST.exists():
        v3_manifest = load_json(BENCHMARK_MANIFEST)
        print(f"Step 1: Loaded benchmark_v3 manifest ({v3_manifest.get('total_records', 'unknown')} records)")
    else:
        print(f"Step 1: benchmark_v3 manifest not found at {BENCHMARK_MANIFEST}")
        v3_manifest = {}

    if INTERNAL_REAL_MANIFEST.exists():
        internal_manifest = load_json(INTERNAL_REAL_MANIFEST)
        candidates = internal_manifest.get("candidates", [])
        print(f"Step 2: Loaded internal_real_v3 manifest ({len(candidates)} candidates)")
    else:
        print(f"Step 2: internal_real_v3 manifest not found at {INTERNAL_REAL_MANIFEST}")
        candidates = []

    # Filter eligible candidates: must have required fields for review
    eligible = []
    for c in candidates:
        has_source = bool(c.get("source"))
        has_license = bool(c.get("license") or c.get("privacy_status"))
        has_quality = bool(c.get("scan_quality"))
        has_elements = bool(c.get("expected_element_groups"))
        if has_source and has_license and has_quality and has_elements:
            eligible.append(c)

    print(f"Step 3: Filtered to {len(eligible)} eligible candidates (have source/license/quality/elements)")

    # Select top candidates for double-review
    # Prioritize: diverse sources, varying quality, high element coverage
    def review_score(c):
        quality = c.get("scan_quality", "unknown")
        quality_weight = {"high": 1.0, "medium": 0.8, "low": 0.6, "unknown": 0.3}.get(quality, 0.5)
        element_count = len(c.get("expected_element_groups", []))
        element_weight = min(element_count / 5.0, 1.0)
        # Prioritize low/medium quality for review (more informative)
        difficulty_bonus = 1.0 if quality in ("low", "medium") else 0.5
        return quality_weight * 0.3 + element_weight * 0.3 + difficulty_bonus * 0.4

    eligible.sort(key=review_score, reverse=True)

    # Ensure source diversity: take top from each source
    source_groups = {}
    for c in eligible:
        src = c.get("source", "unknown")
        source_groups.setdefault(src, []).append(c)

    selected = []
    per_source_target = max(TARGET_LOCKED // max(len(source_groups), 1), 10)
    for src, group in source_groups.items():
        selected.extend(group[:per_source_target])

    # If still not enough, fill from remaining
    selected_ids = {c.get("image_path") or c.get("id") for c in selected}
    for c in eligible:
        cid = c.get("image_path") or c.get("id")
        if cid not in selected_ids and len(selected) < TARGET_LOCKED:
            selected.append(c)
            selected_ids.add(cid)

    selected = selected[:TARGET_LOCKED]
    print(f"Step 4: Selected {len(selected)} candidates for double-review from {len(source_groups)} sources")

    # Generate review queue entries
    review_queue = []
    for i, c in enumerate(selected):
        entry = {
            "review_id": f"R0-T3-LK-{i+1:04d}",
            "image_path": c.get("image_path", ""),
            "source": c.get("source", "unknown"),
            "scan_quality": c.get("scan_quality", "unknown"),
            "expected_element_groups": c.get("expected_element_groups", []),
            "license": c.get("license", ""),
            "privacy_status": c.get("privacy_status", ""),
            "uncertainty_tags": [],
            "conflict_status": "pending",
            "reviewer_1": "",
            "reviewer_2": "",
            "reviewer_1_result": {},
            "reviewer_2_result": {},
            "final_labels": {},
            "notes": ""
        }
        review_queue.append(entry)

    # Create output directory
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Write review_queue.jsonl
    jsonl_path = REVIEW_DIR / "review_queue.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in review_queue:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Step 5: Wrote {jsonl_path} ({len(review_queue)} entries)")

    # Write review_queue.csv
    csv_path = REVIEW_DIR / "review_queue.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "review_id", "image_path", "source", "scan_quality",
            "expected_element_groups", "license", "privacy_status",
            "conflict_status", "notes"
        ])
        for entry in review_queue:
            writer.writerow([
                entry["review_id"],
                entry["image_path"],
                entry["source"],
                entry["scan_quality"],
                json.dumps(entry["expected_element_groups"], ensure_ascii=False),
                entry["license"],
                entry["privacy_status"],
                entry["conflict_status"],
                entry["notes"]
            ])
    print(f"Step 6: Wrote {csv_path}")

    # Write review.html
    html_path = REVIEW_DIR / "review.html"
    source_counts = {}
    for e in review_queue:
        src = e["source"]
        source_counts[src] = source_counts.get(src, 0) + 1

    quality_counts = {}
    for e in review_queue:
        q = e["scan_quality"]
        quality_counts[q] = quality_counts.get(q, 0) + 1

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>R0-T3 Double-Review Locked Pack ({len(review_queue)} records)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 14px; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .high {{ background-color: #dff0d8; }}
        .medium {{ background-color: #fcf8e3; }}
        .low {{ background-color: #f2dede; }}
    </style>
</head>
<body>
    <h1>R0-T3: Double-Review Locked Record Pack</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total records:</strong> {len(review_queue)}</p>
        <p><strong>Sources:</strong> {', '.join(f'{k} ({v})' for k, v in source_counts.items())}</p>
        <p><strong>Quality distribution:</strong> {', '.join(f'{k} ({v})' for k, v in quality_counts.items())}</p>
        <p><strong>Target:</strong> 100 double-review locked records (SCI2 paper requirement)</p>
    </div>
    <h2>Review Queue</h2>
    <table>
        <tr>
            <th>Review ID</th>
            <th>Image Path</th>
            <th>Source</th>
            <th>Quality</th>
            <th>Element Groups</th>
            <th>License/Privacy</th>
            <th>Conflict Status</th>
        </tr>
"""
    for entry in review_queue:
        quality_class = entry["scan_quality"]
        html_content += f"""        <tr class="{quality_class}">
            <td>{entry['review_id']}</td>
            <td>{entry['image_path']}</td>
            <td>{entry['source']}</td>
            <td>{entry['scan_quality']}</td>
            <td>{', '.join(entry['expected_element_groups'])}</td>
            <td>{entry.get('license', '') or entry.get('privacy_status', '')}</td>
            <td>{entry['conflict_status']}</td>
        </tr>
"""
    html_content += """    </table>
    <h2>Review Instructions</h2>
    <ol>
        <li>Each record must be reviewed independently by two annotators.</li>
        <li>Record element labels, bounding boxes, and room types for each image.</li>
        <li>If reviewers disagree, mark conflict_status as "conflict" and add notes.</li>
        <li>Final labels are resolved by consensus or third-party adjudication.</li>
    </ol>
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Step 7: Wrote {html_path}")

    # Write audit
    audit = {
        "version": "review_pack_v2",
        "created": "2026-05-01",
        "total_candidates": len(candidates),
        "eligible_candidates": len(eligible),
        "selected_for_review": len(review_queue),
        "target_locked": TARGET_LOCKED,
        "sources": source_counts,
        "quality_distribution": quality_counts,
        "review_status": "pending_human_review",
        "output_files": [
            str(jsonl_path.relative_to(ROOT)),
            str(csv_path.relative_to(ROOT)),
            str(html_path.relative_to(ROOT))
        ],
        "blocking_note": "R0-T3 done_when requires 100 double-review locked records. This pack is ready for human review. Once completed, update review_status to 'complete' and update R0 phase status to 'done'."
    }

    audit_path = REVIEW_DIR / "review_pack_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    print(f"Step 8: Wrote {audit_path}")

    print("=" * 70)
    print(f"Review pack complete: {len(review_queue)} records ready for double-review")
    print(f"Sources: {source_counts}")
    print(f"Quality: {quality_counts}")
    print(f"Output: {REVIEW_DIR}")
    print("=" * 70)

    return audit


def main():
    audit = build_review_queue()
    return 0


if __name__ == "__main__":
    sys.exit(main())

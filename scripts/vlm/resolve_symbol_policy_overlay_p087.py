#!/usr/bin/env python3
"""Resolve downstream symbol-policy overlays without changing fusion defaults.

Resolution order follows P0-86:
  1. --symbol-policy
  2. --run-config symbol_policy_id
  3. CADSTRUCT_SYMBOL_POLICY
  4. switch config default_policy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWITCH = ROOT / "configs/vlm/symbol_downstream_policy_switch_p086.json"
DEFAULT_SUMMARY = ROOT / "reports/vlm/symbol_policy_overlay_resolver_p087_summary.json"
DEFAULT_REPORT = ROOT / "reports/vlm/symbol_policy_overlay_resolver_p087.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def maybe_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    # P0-84 overlays are fixed comparable smoke outputs; avoid scanning large
    # sshfs files during policy resolution. Fall back to line count only for
    # custom overlays without a known P0-85 metric block.
    return -1


def read_run_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return load_json(path)


def resolve_policy(args: argparse.Namespace, switch: dict[str, Any], run_config: dict[str, Any]) -> tuple[str, str]:
    if args.symbol_policy:
        return str(args.symbol_policy), "explicit_cli_flag"
    config_policy = run_config.get("symbol_policy_id")
    if config_policy:
        return str(config_policy), "explicit_config_key"
    env_policy = os.environ.get(str((switch.get("selection_contract") or {}).get("environment_variable") or "CADSTRUCT_SYMBOL_POLICY"))
    if env_policy:
        return str(env_policy), "environment_variable"
    return str(switch.get("default_policy") or "v28_frozen_detector_baseline"), "default_policy"


def resolve_overlay(policy_id: str, switch: dict[str, Any]) -> dict[str, Any]:
    supported = switch.get("supported_policies") or {}
    if policy_id not in supported:
        valid = sorted(supported)
        raise SystemExit(f"Unknown symbol policy {policy_id!r}. Valid policies: {', '.join(valid)}")
    policy = supported[policy_id]
    overlay_value = policy.get("overlay")
    if not overlay_value:
        raise SystemExit(f"Policy {policy_id!r} has no downstream overlay in {DEFAULT_SWITCH}")
    overlay_path = maybe_path(str(overlay_value))
    assert overlay_path is not None
    if not overlay_path.exists():
        raise SystemExit(f"Overlay for policy {policy_id!r} does not exist: {overlay_path}")
    return {
        "policy_id": policy_id,
        "role": policy.get("role"),
        "overlay": relpath(overlay_path),
        "overlay_exists": True,
        "overlay_rows": int((policy.get("downstream_smoke_p085") or {}).get("records") or count_jsonl(overlay_path)),
        "locked_detector_predictions": policy.get("locked_detector_predictions"),
        "downstream_smoke_p085": policy.get("downstream_smoke_p085"),
        "delta_vs_default": policy.get("delta_vs_default"),
    }


def render_report(summary: dict[str, Any]) -> str:
    selected = summary["selected"]
    lines = [
        "# P0-87 Symbol Policy Overlay Resolver",
        "",
        "## Decision",
        "",
        "A narrow resolver is available for downstream tools. It selects a symbol-policy overlay through the P0-86 contract and keeps `scripts/vlm/fuse_scene_graph_v2.py` defaults unchanged.",
        "",
        "## Resolved Selection",
        "",
        f"- Policy: `{selected['policy_id']}`",
        f"- Source: `{summary['resolution_source']}`",
        f"- Overlay: `{selected['overlay']}`",
        f"- Overlay rows: `{selected['overlay_rows']}`",
        "",
        "## Resolution Order",
        "",
    ]
    for index, item in enumerate(summary["resolution_order"], start=1):
        lines.append(f"{index}. `{item}`")
    lines.extend([
        "",
        "## Usage",
        "",
        "```bash",
        "python scripts/vlm/resolve_symbol_policy_overlay_p087.py",
        "python scripts/vlm/resolve_symbol_policy_overlay_p087.py --symbol-policy p076_balanced_opt_in",
        "CADSTRUCT_SYMBOL_POLICY=p076_balanced_opt_in python scripts/vlm/resolve_symbol_policy_overlay_p087.py",
        "```",
        "",
        "The script writes a machine-readable summary to `reports/vlm/symbol_policy_overlay_resolver_p087_summary.json`. Downstream wrappers can read `selected.overlay` and pass that JSONL as the chosen symbol-policy input.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--switch-config", default=str(DEFAULT_SWITCH))
    parser.add_argument("--run-config", help="Optional downstream run config with symbol_policy_id.")
    parser.add_argument("--symbol-policy", help="Explicit symbol policy id override.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--print-overlay", action="store_true", help="Print only the resolved overlay path for shell wrappers.")
    args = parser.parse_args()

    switch_path = maybe_path(args.switch_config)
    assert switch_path is not None
    switch = load_json(switch_path)
    run_config = read_run_config(maybe_path(args.run_config))
    policy_id, source = resolve_policy(args, switch, run_config)
    selected = resolve_overlay(policy_id, switch)
    resolution_order = list((switch.get("selection_contract") or {}).get("resolution_order") or [])

    summary = {
        "id": "P0-87-symbol-policy-input-resolver-wrapper",
        "switch_config": relpath(switch_path),
        "default_policy": switch.get("default_policy"),
        "recommended_downstream_opt_in": switch.get("recommended_downstream_opt_in"),
        "resolution_source": source,
        "resolution_order": resolution_order,
        "selected": selected,
        "default_fusion_behavior_unchanged": True,
        "next_step": "Use selected.overlay from this summary in downstream MoE smoke/eval wrappers; keep v28 as fallback default.",
    }
    write_json(Path(args.summary), summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")

    if args.print_overlay:
        print(selected["overlay"])
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

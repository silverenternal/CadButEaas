#!/usr/bin/env python3
"""Check P113 full symbol eval async run status."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / 'reports/vlm/full_symbol_eval_p113_logs'


def latest_status() -> Path | None:
    files = sorted(LOG_DIR.glob('*.status.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def tail(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    data = path.read_text(errors='replace').splitlines()
    return data[-lines:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', help='Specific status json; defaults to latest P113 status.')
    parser.add_argument('--tail', type=int, default=20)
    args = parser.parse_args()
    status_path = Path(args.status) if args.status else latest_status()
    if status_path is None:
        raise SystemExit('No P113 status files found.')
    if not status_path.is_absolute():
        status_path = ROOT / status_path
    data = json.loads(status_path.read_text())
    pid = data.get('pid')
    running = False
    if isinstance(pid, int):
        try:
            os.kill(pid, 0)
            running = True
        except OSError:
            running = False
    data['process_running_now'] = running
    if data.get('state') == 'running' and not running:
        eval_rel = data.get('eval_output', '')
        pred_rel = data.get('predictions_output', '')
        eval_exists = bool(eval_rel and (ROOT / eval_rel).exists())
        pred_exists = bool(pred_rel and (ROOT / pred_rel).exists())
        if not eval_exists and not pred_exists:
            data['derived_state'] = 'stale_or_crashed_no_outputs'
        else:
            data['derived_state'] = 'finished_without_status_update'
    log_path = ROOT / data.get('log_file', '')
    eval_path = ROOT / data.get('eval_output', '')
    pred_path = ROOT / data.get('predictions_output', '')
    data['eval_output_exists'] = eval_path.exists()
    data['predictions_output_exists'] = pred_path.exists()
    if eval_path.exists():
        data['eval_output_bytes'] = eval_path.stat().st_size
    if pred_path.exists():
        data['predictions_output_bytes'] = pred_path.stat().st_size
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print('\n--- log tail ---')
    for line in tail(log_path, args.tail):
        print(line)


if __name__ == '__main__':
    main()

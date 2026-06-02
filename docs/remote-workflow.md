# Remote Workflow

This project is usually edited locally through the SSHFS mount and executed on the remote GPU/server.
Use `scripts/remote_ctl.sh` as the default interaction wrapper.

## Defaults

- SSH target: `hugo@47.110.35.232:33022`
- Project dir: `/home/hugo/codes/CadButEaas`
- Python: `/home/hugo/codes/CadButEaas/.venv/bin/python`
- Remote run logs: `/home/hugo/codes/CadButEaas/logs/remote-runs`
- SSH reliability: keepalive, connection timeout, retry, and optional SSH multiplexing are enabled by default.

Override with environment variables when needed:

```bash
REMOTE_HOST=47.110.35.232 REMOTE_PORT=33022 scripts/remote_ctl.sh ping
CUDA_VISIBLE_DEVICES=0 scripts/remote_ctl.sh start-py smoke scripts/vlm/run_cadstruct_moe_smoke_v18.py
REMOTE_RETRIES=5 REMOTE_RETRY_DELAY=3 scripts/remote_ctl.sh run -- nvidia-smi
```

Optional remote environment variables can be stored in `.remote_env` on the remote project root.
The script sources this file before every run.

## Common Commands

Check connectivity:

```bash
scripts/remote_ctl.sh ping
scripts/remote_ctl.sh doctor
```

Run a short foreground command and stream output locally:

```bash
scripts/remote_ctl.sh run -- nvidia-smi
scripts/remote_ctl.sh py scripts/vlm/run_cadstruct_moe_smoke_v18.py
scripts/remote_ctl.sh cargo test -p common-types
```

Start a long background job:

```bash
scripts/remote_ctl.sh start moe-smoke -- python scripts/vlm/image_only_moe_v17_pipeline.py run-all --limit 64
scripts/remote_ctl.sh start-py smoke scripts/vlm/run_cadstruct_moe_smoke_v18.py
```

Read output from a background job:

```bash
scripts/remote_ctl.sh tail latest
scripts/remote_ctl.sh tail latest -f
scripts/remote_ctl.sh status latest
scripts/remote_ctl.sh list
```

Stop a background job:

```bash
scripts/remote_ctl.sh stop latest
scripts/remote_ctl.sh stop moe-smoke
```

Fetch generated files/directories to local `remote-outputs/`:

```bash
scripts/remote_ctl.sh fetch reports/vlm/some_report.json
scripts/remote_ctl.sh fetch logs/remote-runs
```

## Notes

- Foreground runs also write logs under `logs/remote-runs/manual-*.log`.
- Background run names are sanitized and stored as timestamped log/meta/pid files.
- Old remote `.runner.sh` files older than 7 days and `.log` files older than 30 days are cleaned opportunistically.
- SSH keepalive, retries, and ControlMaster multiplexing are enabled in the wrapper.
- The SSHFS mount is maintained separately by the local `cadbut-sshfs-watchdog.timer`, which checks every 30 seconds and remounts stale mounts.

## Local SSHFS Watchdog

The local mount watchdog lives outside the repository:

```bash
systemctl --user status cadbut-sshfs-watchdog.timer
systemctl --user start cadbut-sshfs-watchdog.service
tail -f ~/.local/state/cadbut-sshfs-watchdog.log
```

It verifies both SSH reachability and mounted-directory responsiveness before remounting.

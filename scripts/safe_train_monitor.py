#!/usr/bin/env python3
"""Run training while logging GPU health and stopping sustained overheating."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=90)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        except Exception:
            os.killpg(process.pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-temp", type=int, default=87)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("a command is required after --")

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    console_path = run_dir / "training_console.log"
    health_path = run_dir / "gpu_health.csv"
    stop_event = threading.Event()
    guard_reason: list[str] = []

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        start_new_session=True,
    )

    def monitor() -> None:
        hot_count = 0
        with health_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(["time_utc", "temperature_c", "power_w", "memory_used_mib", "memory_total_mib", "free_disk_gib"])
            while not stop_event.wait(args.poll_seconds):
                if process.poll() is not None:
                    return
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,memory.used,memory.total", "--format=csv,noheader,nounits"],
                        check=True, capture_output=True, text=True,
                    )
                    temp, power, used, total = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
                    free_gib = shutil.disk_usage(run_dir).free / 1024**3
                    writer.writerow([datetime.now(timezone.utc).isoformat(), temp, power, used, total, f"{free_gib:.2f}"])
                    handle.flush()
                    hot_count = hot_count + 1 if float(temp) >= args.max_temp else 0
                    if hot_count >= 3:
                        guard_reason.append(f"GPU stayed at or above {args.max_temp} C for three checks")
                    elif free_gib < args.min_free_gib:
                        guard_reason.append(f"free disk dropped below {args.min_free_gib:.1f} GiB")
                    if guard_reason:
                        stop_process(process)
                        return
                except Exception as exc:
                    writer.writerow([datetime.now(timezone.utc).isoformat(), "monitor_error", str(exc), "", "", ""])
                    handle.flush()

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    saw_oom = False
    try:
        assert process.stdout is not None
        recent_output = b""
        with console_path.open("ab") as log:
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                # Preserve carriage returns so tqdm updates one terminal line
                # instead of becoming hundreds of newline-delimited messages.
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log.write(chunk)
                log.flush()
                recent_output = (recent_output + chunk.lower())[-4096:]
                saw_oom = saw_oom or b"out of memory" in recent_output
        return_code = process.wait()
    except KeyboardInterrupt:
        guard_reason.append("interrupted by user")
        stop_process(process)
        return_code = 130
    finally:
        stop_event.set()
        worker.join(timeout=5)

    if guard_reason:
        print(f"SAFETY STOP: {guard_reason[0]}", file=sys.stderr)
        print("The last completed epoch remains in latest.pth and will resume automatically.", file=sys.stderr)
        return 3
    if saw_oom:
        print("CUDA OOM detected. Close GPU applications and retry; automatic resume is enabled.", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

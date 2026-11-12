#!/usr/bin/env python3
"""Fail-fast checks for the ClaimGuard FUSED WSL training profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED_CHECKPOINT_SHA256 = "7e527d49080a773641cf96915b4b3fa12d571c5b87f35c527f13ab82f148364c"
MIN_VRAM_GIB = 7.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nvidia_smi() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,driver_version,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    first = result.stdout.strip().splitlines()[0]
    name, memory_mib, used_mib, driver, temperature = [part.strip() for part in first.split(",", 4)]
    return {
        "name": name,
        "memory_gib": round(float(memory_mib) / 1024, 2),
        "memory_used_gib": round(float(used_mib) / 1024, 2),
        "driver": driver,
        "temperature_c": int(float(temperature)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("host", "train"), default="train")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {
        "project_root": str(root),
        "stage": args.stage,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    if sys.version_info < (3, 10) or sys.version_info >= (3, 13):
        failures.append("Python 3.10-3.12 is required (3.11 recommended)")

    try:
        kernel = Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        kernel = ""
    if "microsoft" not in kernel:
        failures.append("this launcher is intended to run inside WSL2")
    if str(root).startswith("/mnt/"):
        warnings.append("project is under /mnt/*; move it to /home/* for much faster image I/O")

    free_gib = shutil.disk_usage(root).free / 1024**3
    report["free_disk_gib"] = round(free_gib, 2)
    required_free_gib = 20.0 if args.stage == "host" else 12.0
    if free_gib < required_free_gib:
        failures.append(f"at least {required_free_gib:.0f} GiB free disk is required; found {free_gib:.1f} GiB")

    try:
        memory_gib = int(Path("/proc/meminfo").read_text().split("MemTotal:", 1)[1].split()[0]) / 1024**2
        report["system_memory_gib"] = round(memory_gib, 2)
        if memory_gib < 12:
            failures.append(f"at least 12 GiB system RAM is required; found {memory_gib:.1f} GiB")
        elif memory_gib < 16:
            warnings.append("less than 16 GiB RAM detected; close memory-heavy applications")
    except Exception:
        warnings.append("could not read total system RAM")

    power_files = list(Path("/sys/class/power_supply").glob("*/online"))
    if power_files:
        power_values = [path.read_text().strip() for path in power_files]
        if "1" not in power_values:
            failures.append("AC power is not connected; laptop training on battery is blocked")

    checkpoint = root / "weights" / "fused" / "fused_opensdi.pth"
    if not checkpoint.is_file():
        failures.append(f"missing checkpoint: {checkpoint}")
    elif checkpoint.stat().st_size < 300_000_000:
        failures.append(f"checkpoint looks truncated: {checkpoint.stat().st_size} bytes")
    else:
        actual = sha256(checkpoint)
        report["checkpoint_sha256"] = actual
        if actual != EXPECTED_CHECKPOINT_SHA256:
            failures.append("FUSED checkpoint SHA-256 does not match the supplied model")

    try:
        gpu = nvidia_smi()
        report["gpu"] = gpu
        if float(gpu["memory_gib"]) < MIN_VRAM_GIB:
            failures.append(
                f"full FUSED safe profile needs at least {MIN_VRAM_GIB:.0f} GiB VRAM; "
                f"found {gpu['memory_gib']} GiB"
            )
        if float(gpu["memory_used_gib"]) > 1.5:
            failures.append(
                f"{gpu['memory_used_gib']} GiB VRAM is already in use; close GPU applications first"
            )
        if int(gpu["temperature_c"]) >= 85:
            failures.append(f"GPU is already too hot to start ({gpu['temperature_c']} C)")
    except Exception as exc:
        failures.append(f"nvidia-smi / WSL GPU passthrough unavailable: {exc}")

    if args.stage == "train":
        required = ("torch", "torchvision", "timm", "cv2", "sklearn", "hydra", "albumentations")
        for module in required:
            try:
                __import__(module)
            except Exception as exc:
                failures.append(f"Python dependency unavailable: {module} ({exc})")
        try:
            import torch

            report["torch"] = torch.__version__
            report["torch_cuda"] = torch.version.cuda
            if not torch.cuda.is_available():
                failures.append("torch.cuda.is_available() is false")
            elif not torch.cuda.is_bf16_supported():
                failures.append("GPU does not support BF16 required by the laptop-safe full FUSED profile")
        except Exception:
            pass

        cache = Path(os.environ.get("HF_HOME", root / "downloads" / "huggingface"))
        cached_bytes = sum(path.stat().st_size for path in cache.rglob("*") if path.is_file()) if cache.exists() else 0
        report["huggingface_cache"] = str(cache)
        report["huggingface_cache_gib"] = round(cached_bytes / 1024**3, 2)
        if cached_bytes < 1_000_000_000:
            failures.append("ConvNeXt-XXL cache is missing; run scripts/setup_fused_wsl.sh first")

    report["warnings"] = warnings
    report["failures"] = failures
    report["ok"] = not failures
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("ClaimGuard FUSED preflight")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

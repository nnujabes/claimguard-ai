#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 0 ]]; then
  echo "No argument is needed. Put images in: $ROOT/data/fused_source" >&2
  exit 2
fi

if [[ ! -x "$ROOT/.venvs/fused/bin/python" ]]; then
  bash "$ROOT/scripts/setup_fused_wsl.sh"
fi
bash "$ROOT/scripts/run_fused_finetune_wsl.sh"

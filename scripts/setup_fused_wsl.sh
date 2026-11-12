#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIP_DISABLE_PIP_VERSION_CHECK=1
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

for candidate in python3.11 python3.12 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    HOST_PYTHON="$candidate"
    break
  fi
done
if [[ -z "${HOST_PYTHON:-}" ]]; then
  echo "Python 3.10-3.12 is required." >&2
  exit 2
fi

DEFAULT_HF_HOME="$("$HOST_PYTHON" -c 'from pathlib import Path; print(Path.home()/".cache"/"huggingface")')"
if [[ -d "$DEFAULT_HF_HOME/hub/models--timm--convnext_xxlarge.clip_laion2b_soup" ]]; then
  export HF_HOME="$DEFAULT_HF_HOME"
else
  export HF_HOME="$ROOT/downloads/huggingface"
fi
echo "Using Hugging Face cache: $HF_HOME"

"$HOST_PYTHON" "$ROOT/scripts/preflight_fused_wsl.py" --project-root "$ROOT" --stage host

if [[ ! -x "$ROOT/.venvs/fused/bin/python" ]]; then
  "$HOST_PYTHON" -m venv "$ROOT/.venvs/fused"
fi
PYTHON="$ROOT/.venvs/fused/bin/python"
PIP="$ROOT/.venvs/fused/bin/pip"

"$PIP" install --upgrade pip setuptools wheel
"$PIP" install torch==2.9.1 torchvision==0.24.1 --index-url "$TORCH_INDEX_URL"
"$PIP" install -e "$ROOT/vendor/fused"

mkdir -p "$HF_HOME"
echo "Caching the frozen ConvNeXt-XXL backbone (one-time download)..."
"$PYTHON" - <<'PY'
import gc
import timm
model = timm.create_model("convnext_xxlarge.clip_laion2b_soup", pretrained=True, num_classes=0)
print("ConvNeXt-XXL cache ready")
del model
gc.collect()
PY

"$PYTHON" "$ROOT/scripts/preflight_fused_wsl.py" --project-root "$ROOT" --stage train
echo "Setup complete. Put images in data/fused_source, then run: bash scripts/run_fused_finetune_wsl.sh"

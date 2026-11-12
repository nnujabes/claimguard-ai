#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 0 ]]; then
  echo "No argument is needed. Put images in: $ROOT/data/fused_source" >&2
  exit 2
fi

DATA_INPUT="$ROOT/data/fused_source/data/data/raw_by_part"
PYTHON="$ROOT/.venvs/fused/bin/python"
MANIFEST_DIR="$ROOT/data/fused_manifests"
RUN_DIR="${CG_FUSED_RUN_DIR:-$ROOT/runs/fused_claimguard_30ep}"
CHECKPOINT="$ROOT/weights/fused/fused_opensdi.pth"

if [[ ! -x "$PYTHON" ]]; then
  echo "Run this first: bash scripts/setup_fused_wsl.sh" >&2
  exit 2
fi
if [[ ! -d "$DATA_INPUT" ]]; then
  echo "Training image directory is missing: $DATA_INPUT" >&2
  exit 2
fi

DEFAULT_HF_HOME="$("$PYTHON" -c 'from pathlib import Path; print(Path.home()/".cache"/"huggingface")')"
if [[ -d "$DEFAULT_HF_HOME/hub/models--timm--convnext_xxlarge.clip_laion2b_soup" ]]; then
  export HF_HOME="$DEFAULT_HF_HOME"
else
  export HF_HOME="$ROOT/downloads/huggingface"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

"$PYTHON" "$ROOT/scripts/preflight_fused_wsl.py" --project-root "$ROOT" --stage train
"$PYTHON" "$ROOT/scripts/prepare_fused_dataset.py" \
  --data "$DATA_INPUT" \
  --output-dir "$MANIFEST_DIR" \
  --seed 42

mkdir -p "$RUN_DIR"
cp "$MANIFEST_DIR/dataset_report.json" "$RUN_DIR/dataset_report.json"

RESUME_OVERRIDE="trainer.resume=null"
if [[ -s "$RUN_DIR/latest.pth" ]]; then
  RESUME_OVERRIDE="trainer.resume=$RUN_DIR/latest.pth"
  echo "Resuming from $RUN_DIR/latest.pth"
fi

COMMAND=(
  "$PYTHON" "$ROOT/vendor/fused/train.py"
  "trainer.epochs=30"
  "trainer.batch_size=1"
  "trainer.accum=8"
  "trainer.lr=3.0e-5"
  "trainer.weight_decay=1.0e-4"
  "trainer.num_workers=2"
  "trainer.log_every=25"
  "trainer.verbose=1"
  "trainer.grad_clip=1.0"
  "trainer.deterministic=true"
  "trainer.frozen_backbone_bf16=true"
  "trainer.checkpoint=$CHECKPOINT"
  "trainer.out_dir=$RUN_DIR"
  "$RESUME_OVERRIDE"
  "dataset.train.manifest_path=$MANIFEST_DIR/train.json"
  "dataset.train.augment=true"
  "dataset.val.manifest_path=$MANIFEST_DIR/val.json"
  "dataset.val.augment=false"
  "dataset.test.manifest_path=$MANIFEST_DIR/test.json"
  "dataset.test.augment=false"
  "wandb.mode=disabled"
  "model.enable_sparsevit_branch=true"
  "model.enable_convnext_branch=true"
)

printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/launch_command.txt"
printf '\n' >> "$RUN_DIR/launch_command.txt"

"$PYTHON" "$ROOT/scripts/safe_train_monitor.py" \
  --run-dir "$RUN_DIR" \
  --max-temp "${CG_MAX_GPU_TEMP:-87}" \
  --min-free-gib 5 \
  --poll-seconds 30 \
  -- "${COMMAND[@]}"

DEPLOY_DIR="$ROOT/weights/fused/finetuned"
mkdir -p "$DEPLOY_DIR"
cp "$RUN_DIR/best.pth" "$DEPLOY_DIR/latest.pth.tmp"
mv "$DEPLOY_DIR/latest.pth.tmp" "$DEPLOY_DIR/latest.pth"
cp "$RUN_DIR/config.yaml" "$DEPLOY_DIR/config.yaml"
cp "$RUN_DIR/history.json" "$DEPLOY_DIR/history.json"

echo "Training complete: $RUN_DIR/best.pth"
echo "Deployed as the app checkpoint: $DEPLOY_DIR/latest.pth"

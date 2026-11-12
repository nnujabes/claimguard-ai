#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.visualization import save_heatmaps

REPO = ROOT / "vendor/fused"
sys.path.insert(0, str(REPO))
from eval.opensdi import build_model


class FusedInferenceEngine:
    """Load one FUSED checkpoint once and reuse it for many images."""

    def __init__(self, checkpoint: str | Path, bf16_backbone: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_FAIL: torch.cuda.is_available() is false")
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device("cuda")
        self.model = build_model(str(self.checkpoint), self.device, True)
        self.bf16_backbone = bool(bf16_backbone) and hasattr(self.model, "convnext")
        if self.bf16_backbone:
            self.model.convnext.to(dtype=torch.bfloat16)
        self.autocast_dtype = torch.bfloat16 if self.bf16_backbone else torch.float16
        self.transform = transforms.Compose([
            transforms.Resize([512, 512]), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        raw = torch.load(self.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        state = raw.get("state_dict", raw.get("model", raw))
        state = {key.removeprefix("module."): value for key, value in state.items()}
        model_keys, checkpoint_keys = set(self.model.state_dict()), set(state)
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        has_convnext = any(key.startswith("convnext.") for key in checkpoint_keys)
        self.audit = {
            "matched_keys": len(checkpoint_keys) - len(unexpected),
            "missing_required_keys": [
                key for key in missing
                if not key.startswith("convnext.") and not key.endswith("num_batches_tracked")
            ],
            "missing_frozen_convnext_keys": sum(key.startswith("convnext.") for key in missing),
            "unexpected_checkpoint_keys": unexpected,
            "frozen_semantic_backbone_loaded_by_timm": not has_convnext,
        }
        self.semantic_backbone_source = "checkpoint" if has_convnext else "timm_pretrained"
        del raw, state
        gc.collect()

    def infer(self, image_path: str | Path, output_dir: str | Path, save_visuals: bool = True) -> dict:
        started = time.perf_counter()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        pil = Image.open(image_path).convert("RGB"); width, height = pil.size
        input_tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.autocast_dtype):
            pred = self.model(input_tensor)
        native_prob = pred["logits"].sigmoid()[0, 0].float().cpu().numpy()
        score = float(pred["cls_logits"].softmax(-1)[0, 1].item())
        if not np.isfinite(score) or not 0 <= score <= 1 or not np.isfinite(native_prob).all():
            raise RuntimeError("FUSED returned NaN/Inf or an out-of-range score")
        result = {
            "model": "fused", "success": True, "score": score,
            "score_semantics": "higher_means_more_manipulated",
            "score_source": "softmax(cls_logits)[fake_class_index=1]",
            "input_tensor_shape": list(input_tensor.shape), "input_color_mode": "RGB",
            "preprocessing": "Resize(512,512)+ToTensor+ImageNetNormalize",
            "checkpoint_path": str(self.checkpoint), "checkpoint_load_audit": self.audit,
            "semantic_backbone_source": self.semantic_backbone_source,
            "prob_map_path": None, "mask_path": None, "overlay_path": None, "map_path": None,
            "heatmap_path": None, "heatmap_overlay_path": None,
            "device": "cuda", "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1048576, 2),
            "inference_sec": round(time.perf_counter() - started, 6),
            "save_visuals": bool(save_visuals),
        }
        if save_visuals:
            prob = np.asarray(Image.fromarray((native_prob * 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            mask = (prob >= .5).astype(np.uint8)
            probability_path, mask_path, overlay_path = (
                out / "probability_map.png", out / "binary_mask.png", out / "overlay.png")
            Image.fromarray((prob * 255).astype(np.uint8)).save(probability_path)
            Image.fromarray(mask * 255).save(mask_path)
            base = np.asarray(pil, dtype=np.float32); red = np.zeros_like(base); red[..., 0] = 255
            alpha = mask[..., None].astype(np.float32) * .5
            Image.fromarray((base*(1-alpha)+red*alpha).clip(0,255).astype(np.uint8)).save(overlay_path)
            map_path = out / "localization_map.npy"; np.save(map_path, native_prob)
            result.update({"prob_map_path":str(probability_path.resolve()),"mask_path":str(mask_path.resolve()),
                           "overlay_path":str(overlay_path.resolve()),"map_path":str(map_path.resolve())})
            result.update(save_heatmaps(image_path, native_prob, out))
        return result


def failure(exc: Exception) -> dict:
    return {"model": "fused", "success": False,
            "error_code": "GPU_FAIL" if "GPU_FAIL" in str(exc) else "FUSED_INFERENCE_FAILED",
            "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--output-dir")
    parser.add_argument("--result-json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--no-visuals", action="store_true")
    args = parser.parse_args()
    bf16_backbone = os.getenv("CG_FUSED_BF16", "0").lower() in {"1", "true", "yes"}
    try:
        engine = FusedInferenceEngine(args.checkpoint, bf16_backbone=bf16_backbone)
    except Exception as exc:
        traceback.print_exc()
        if args.serve:
            print(json.dumps({"kind": "ready", "success": False, "error": str(exc)}), flush=True)
        elif args.result_json:
            Path(args.result_json).write_text(json.dumps(failure(exc), indent=2) + "\n")
        return 1

    if args.serve:
        print(json.dumps({"kind": "ready", "success": True,
                          "checkpoint": str(engine.checkpoint),
                          "bf16_backbone": engine.bf16_backbone,
                          "resident_vram_mb": round(torch.cuda.memory_allocated() / 1048576, 2)}), flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                result = engine.infer(request["image"], request["output_dir"],
                                      save_visuals=bool(request.get("save_visuals", True)))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr); torch.cuda.empty_cache(); result = failure(exc)
            print(json.dumps({"kind": "result", "result": result}), flush=True)
        return 0

    if not all((args.image, args.output_dir, args.result_json)):
        parser.error("--image, --output-dir and --result-json are required without --serve")
    try:
        result = engine.infer(args.image, args.output_dir, save_visuals=not args.no_visuals)
    except Exception as exc:
        traceback.print_exc(); result = failure(exc)
    result_path = Path(args.result_json); result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

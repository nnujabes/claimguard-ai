#!/usr/bin/env python3
"""GPU-only adapter for UserPollo/moire-pattern-detector."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ROOT / "weights/recapture"
BACKBONE_DIR = ROOT / "downloads/recapture_dinov2/dinov2-with-registers-base"


class ScreenDetectorMLP(nn.Module):
    """Exact classification head published in the upstream model card."""
    def __init__(self, input_size=3072, hidden_size=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.GELU(),
            nn.BatchNorm1d(hidden_size), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, value):
        return self.mlp(value)


def normalize_classes(raw):
    if isinstance(raw, list):
        return {str(i): str(label) for i, label in enumerate(raw)}
    return {str(key): str(value) for key, value in raw.items()}


class RecaptureInferenceEngine:
    """Load the DINOv2 backbone and classifier once for repeated requests."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_FAIL: torch.cuda.is_available() is false")
        self.device = torch.device("cuda")
        self.classes = normalize_classes(json.loads((CHECKPOINTS / "classes.json").read_text()))
        moire_indices = [int(index) for index, label in self.classes.items()
                         if label.lower() == "moire"]
        if len(moire_indices) != 1:
            raise RuntimeError(f"classes.json must identify exactly one moire class: {self.classes}")
        self.moire_index = moire_indices[0]
        self.backbone = AutoModel.from_pretrained(
            BACKBONE_DIR, local_files_only=True).to(self.device).eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        backbone_state = torch.load(CHECKPOINTS / "best_screen_detector_backbone.pt",
                                    map_location=self.device, weights_only=True)
        total_layers = len(self.backbone.encoder.layer)
        for offset, layer in enumerate(self.backbone.encoder.layer[total_layers - 2:]):
            layer.load_state_dict(backbone_state[f"layer.{total_layers - 2 + offset}"])
        self.head = ScreenDetectorMLP(input_size=3072, num_classes=len(self.classes)).to(self.device)
        self.head.load_state_dict(torch.load(CHECKPOINTS / "best_screen_detector_mlp.pt",
                                             map_location=self.device, weights_only=True))
        self.head.eval()
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.local_transform = transforms.Compose([
            transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.global_transform = transforms.Compose([
            transforms.Resize((256, 256)), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(mean, std)])

    def infer(self, image_path):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        image = Image.open(image_path).convert("RGB")
        batch = torch.stack([self.local_transform(image), self.global_transform(image)]).to(self.device)
        with torch.inference_mode():
            hidden = self.backbone(pixel_values=batch).last_hidden_state.float()
            register_count = getattr(self.backbone.config, "num_register_tokens", 0)
            cls_token = hidden[:, 0, :]
            patch_mean = hidden[:, 1 + register_count:, :].mean(dim=1)
            branch_features = torch.cat([cls_token, patch_mean], dim=-1)
            combined = torch.cat([branch_features[0:1], branch_features[1:2]], dim=-1)
            probabilities = torch.softmax(self.head(combined), dim=1)[0]
        score = float(probabilities[self.moire_index].item())
        predicted_index = int(probabilities.argmax().item())
        if not 0.0 <= score <= 1.0 or not torch.isfinite(probabilities).all():
            raise RuntimeError(f"Invalid recapture probabilities: {probabilities.tolist()}")
        return {"model":"recapture","success":True,"score":score,
                "score_semantics":"higher_means_more_screen_recapture_or_moire_like",
                "signal_scope":"screen-recapture / moire-like signal; not AI manipulation probability",
                "score_source":f"softmax(class={self.classes[str(self.moire_index)]}, index={self.moire_index})",
                "class_mapping":self.classes,"predicted_class":self.classes[str(predicted_index)],
                "class_probabilities":{self.classes[str(i)]:float(value)
                                       for i,value in enumerate(probabilities)},
                "device":"cuda","peak_vram_mb":round(torch.cuda.max_memory_allocated(self.device)/1048576,2),
                "inference_sec":round(time.perf_counter()-started,6)}


def failure(exc):
    return {"model":"recapture","success":False,
            "error_code":"GPU_FAIL" if "GPU_FAIL" in str(exc) else "RECAPTURE_INFERENCE_FAILED",
            "error":str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--output-dir")
    parser.add_argument("--result-json")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.serve:
        try:
            engine = RecaptureInferenceEngine()
        except Exception as exc:
            traceback.print_exc()
            print(json.dumps({"kind":"ready","success":False,"error":str(exc)}),flush=True)
            return 1
        print(json.dumps({"kind":"ready","success":True,
                          "resident_vram_mb":round(torch.cuda.memory_allocated(engine.device)/1048576,2)}),flush=True)
        for line in sys.stdin:
            try:
                request=json.loads(line);result=engine.infer(request["image"])
            except Exception as exc:
                traceback.print_exc(file=sys.stderr);torch.cuda.empty_cache();result=failure(exc)
            print(json.dumps({"kind":"result","result":result}),flush=True)
        return 0
    if not all((args.image,args.output_dir,args.result_json)):
        parser.error("--image, --output-dir and --result-json are required without --serve")
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_FAIL: torch.cuda.is_available() is false")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)

        classes = normalize_classes(json.loads((CHECKPOINTS / "classes.json").read_text()))
        moire_indices = [int(index) for index, label in classes.items() if label.lower() == "moire"]
        if len(moire_indices) != 1:
            raise RuntimeError(f"classes.json must identify exactly one moire class: {classes}")
        moire_index = moire_indices[0]

        backbone = AutoModel.from_pretrained(BACKBONE_DIR, local_files_only=True).to(device).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone_state = torch.load(
            CHECKPOINTS / "best_screen_detector_backbone.pt",
            map_location=device, weights_only=True,
        )
        total_layers = len(backbone.encoder.layer)
        for offset, layer in enumerate(backbone.encoder.layer[total_layers - 2:]):
            layer.load_state_dict(backbone_state[f"layer.{total_layers - 2 + offset}"])

        head = ScreenDetectorMLP(input_size=3072, num_classes=len(classes)).to(device)
        head.load_state_dict(torch.load(
            CHECKPOINTS / "best_screen_detector_mlp.pt",
            map_location=device, weights_only=True,
        ))
        head.eval()

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        local_transform = transforms.Compose([
            transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        global_transform = transforms.Compose([
            transforms.Resize((256, 256)), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize(mean, std),
        ])
        image = Image.open(args.image).convert("RGB")
        batch = torch.stack([local_transform(image), global_transform(image)]).to(device)
        with torch.inference_mode():
            hidden = backbone(pixel_values=batch).last_hidden_state.float()
            register_count = getattr(backbone.config, "num_register_tokens", 0)
            cls_token = hidden[:, 0, :]
            patch_mean = hidden[:, 1 + register_count:, :].mean(dim=1)
            branch_features = torch.cat([cls_token, patch_mean], dim=-1)
            combined = torch.cat([branch_features[0:1], branch_features[1:2]], dim=-1)
            probabilities = torch.softmax(head(combined), dim=1)[0]
        score = float(probabilities[moire_index].item())
        predicted_index = int(probabilities.argmax().item())
        if not 0.0 <= score <= 1.0 or not torch.isfinite(probabilities).all():
            raise RuntimeError(f"Invalid recapture probabilities: {probabilities.tolist()}")
        result = {
            "model": "recapture", "success": True, "score": score,
            "score_semantics": "higher_means_more_screen_recapture_or_moire_like",
            "signal_scope": "screen-recapture / moire-like signal; not AI manipulation probability",
            "score_source": f"softmax(class={classes[str(moire_index)]}, index={moire_index})",
            "class_mapping": classes, "predicted_class": classes[str(predicted_index)],
            "class_probabilities": {classes[str(i)]: float(value) for i, value in enumerate(probabilities)},
            "device": "cuda", "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1048576, 2),
        }
    except Exception as exc:
        traceback.print_exc()
        result = {
            "model": "recapture", "success": False,
            "error_code": "GPU_FAIL" if "GPU_FAIL" in str(exc) else "RECAPTURE_INFERENCE_FAILED",
            "error": str(exc),
        }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

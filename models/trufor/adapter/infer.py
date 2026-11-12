#!/usr/bin/env python3
import argparse
import gc
import inspect
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.visualization import save_heatmaps

REPO = ROOT / "vendor/trufor/TruFor_train_test"
MAX_INFERENCE_SIDE = 2048


class TruForInferenceEngine:
    """Load TruFor once and reuse it while preserving the current 2048px policy."""

    def __init__(self, checkpoint):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_FAIL: torch.cuda.is_available() is false")
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device("cuda:0")
        old_cwd = Path.cwd()
        os.chdir(REPO)
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        try:
            from lib.config import config, update_config
            from lib.utils import get_model
            import torch.backends.cudnn as cudnn

            args = SimpleNamespace(gpu=0, experiment="trufor_ph3",
                                   opts=["TEST.MODEL_FILE", str(self.checkpoint)])
            update_config(config, args)
            cudnn.benchmark = config.CUDNN.BENCHMARK
            cudnn.deterministic = config.CUDNN.DETERMINISTIC
            cudnn.enabled = config.CUDNN.ENABLED
            original_load = torch.load
            supports_weights_only = "weights_only" in inspect.signature(original_load).parameters
            def compatible_load(*load_args, **load_kwargs):
                if supports_weights_only:
                    load_kwargs.setdefault("weights_only", False)
                return original_load(*load_args, **load_kwargs)
            torch.load = compatible_load
            try:
                checkpoint_state = torch.load(self.checkpoint, map_location=self.device)
            finally:
                torch.load = original_load
            self.model = get_model(config)
            self.model.load_state_dict(checkpoint_state["state_dict"])
            self.model = self.model.to(self.device).eval()
            del checkpoint_state
            gc.collect()
        finally:
            os.chdir(old_cwd)

    def infer(self, image_path, output_dir, save_visuals=True):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        source_image = Path(image_path).resolve()
        with Image.open(source_image) as opened:
            original_size = opened.size
            if max(original_size) > MAX_INFERENCE_SIDE:
                scale = MAX_INFERENCE_SIDE / max(original_size)
                inference_size = tuple(max(1, round(side * scale)) for side in original_size)
                resampling = getattr(Image, "Resampling", Image)
                rgb = np.array(opened.convert("RGB").resize(inference_size, resampling.LANCZOS))
            else:
                inference_size = original_size
                rgb = np.array(opened.convert("RGB"))
        tensor = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float,
                              device=self.device).unsqueeze(0) / 256.0
        with torch.no_grad():
            pred, _conf, det, _npp = self.model(tensor)
            if det is None:
                raise RuntimeError("TruFor model returned no detection score")
            score = torch.sigmoid(det).item()
            localization_map = F.softmax(torch.squeeze(pred, 0), dim=0)[1].cpu().numpy()
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError(f"Invalid TruFor score: {score}")
        flat_map = localization_map.reshape(-1)
        top5_count = max(1, int(np.ceil(flat_map.size * 0.05)))
        top1_count = max(1, int(np.ceil(flat_map.size * 0.01)))
        result = {"model":"trufor","success":True,"score":score,
                  "score_semantics":"higher_means_more_manipulated",
                  "score_source":"official_global_integrity_detection_score",
                  "trufor_global_score":score,
                  "trufor_map_top5_mean":float(np.partition(flat_map,-top5_count)[-top5_count:].mean()),
                  "trufor_map_top1_mean":float(np.partition(flat_map,-top1_count)[-top1_count:].mean()),
                  "trufor_map_fraction_gt_05":float((flat_map > 0.5).mean()),
                  "trufor_map_max":float(flat_map.max()),
                  "original_image_size":list(original_size),
                  "inference_image_size":list(inference_size),
                  "input_resized_for_memory":inference_size != original_size,
                  "max_inference_side":MAX_INFERENCE_SIDE,"device":"cuda",
                  "peak_vram_mb":round(torch.cuda.max_memory_allocated(self.device)/1048576,2),
                  "inference_sec":round(time.perf_counter()-started,6),
                  "save_visuals":bool(save_visuals),"map_path":None}
        if save_visuals:
            out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
            map_path = out_dir / "localization_map.npy"
            np.save(map_path, localization_map)
            result["map_path"] = str(map_path.resolve())
            result.update(save_heatmaps(image_path, localization_map, out_dir))
        return result


def failure(exc):
    return {"model":"trufor","success":False,
            "error_code":"GPU_FAIL" if "GPU_FAIL" in str(exc) else "TRUFOR_INFERENCE_FAILED",
            "error":str(exc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image")
    p.add_argument("--output-dir")
    p.add_argument("--result-json")
    p.add_argument("--checkpoint", default=str(ROOT / "weights/trufor/finetuned/best.pth.tar"))
    p.add_argument("--serve", action="store_true")
    p.add_argument("--no-visuals", action="store_true")
    a = p.parse_args()
    if a.serve:
        try:
            engine = TruForInferenceEngine(a.checkpoint)
        except Exception as exc:
            traceback.print_exc()
            print(json.dumps({"kind":"ready","success":False,"error":str(exc)}),flush=True)
            return 1
        print(json.dumps({"kind":"ready","success":True,"checkpoint":str(engine.checkpoint),
                          "resident_vram_mb":round(torch.cuda.memory_allocated(engine.device)/1048576,2)}),flush=True)
        for line in sys.stdin:
            try:
                request=json.loads(line)
                result=engine.infer(request["image"],request["output_dir"],
                                    save_visuals=bool(request.get("save_visuals",True)))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr);result=failure(exc)
            finally:
                torch.cuda.empty_cache()
            print(json.dumps({"kind":"result","result":result}),flush=True)
        return 0
    if not all((a.image,a.output_dir,a.result_json)):
        p.error("--image, --output-dir and --result-json are required without --serve")
    result_path = Path(a.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "trufor_output.npz"
    # The official script deliberately skips existing outputs. Use a private
    # per-run path so every smoke invocation performs a real forward pass.
    raw_run = out_dir / f"trufor_output_run_{os.getpid()}.npz"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU_FAIL: torch.cuda.is_available() is false")
        torch.cuda.reset_peak_memory_stats()
        source_image = Path(a.image).resolve()
        with Image.open(source_image) as opened:
            original_size = opened.size
            if max(original_size) > MAX_INFERENCE_SIDE:
                scale = MAX_INFERENCE_SIDE / max(original_size)
                inference_size = tuple(max(1, round(side * scale)) for side in original_size)
                inference_image = out_dir / "trufor_inference_input.png"
                resampling = getattr(Image, "Resampling", Image)
                opened.convert("RGB").resize(inference_size, resampling.LANCZOS).save(inference_image)
                inference_image = inference_image.resolve()
            else:
                inference_size = original_size
                inference_image = source_image
        # PyTorch 2.6 changed torch.load's default; this trusted official checkpoint
        # contains metadata required by the official loader.
        original_load = torch.load
        supports_weights_only = "weights_only" in inspect.signature(original_load).parameters
        def compatible_load(*args, **kwargs):
            if supports_weights_only:
                kwargs.setdefault("weights_only", False)
            else:
                kwargs.pop("weights_only", None)
            return original_load(*args, **kwargs)
        torch.load = compatible_load
        old_cwd, old_argv = Path.cwd(), sys.argv[:]
        os.chdir(REPO)
        sys.path.insert(0, str(REPO))
        sys.argv = ["test.py", "-g", "0", "-in", str(inference_image),
                    "-out", str(raw_run.resolve()), "-exp", "trufor_ph3",
                    "TEST.MODEL_FILE", str(Path(a.checkpoint).resolve())]
        try:
            runpy.run_path(str(REPO / "test.py"), run_name="__main__")
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
        if not raw_run.exists():
            raise RuntimeError("Official TruFor inference did not create its NPZ output")
        raw_run.replace(raw)
        data = np.load(raw)
        score = float(np.asarray(data["score"]).item())
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError(f"Invalid TruFor score: {score}")
        localization_map = np.asarray(data["map"], dtype=np.float32)
        flat_map = localization_map.reshape(-1)
        top5_count = max(1, int(np.ceil(flat_map.size * 0.05)))
        top1_count = max(1, int(np.ceil(flat_map.size * 0.01)))
        map_path = out_dir / "localization_map.npy"
        np.save(map_path, localization_map)
        result = {"model": "trufor", "success": True, "score": score,
                  "score_semantics": "higher_means_more_manipulated",
                  "score_source": "official_global_integrity_detection_score",
                  "trufor_global_score": score,
                  "trufor_map_top5_mean": float(np.partition(flat_map, -top5_count)[-top5_count:].mean()),
                  "trufor_map_top1_mean": float(np.partition(flat_map, -top1_count)[-top1_count:].mean()),
                  "trufor_map_fraction_gt_05": float((flat_map > 0.5).mean()),
                  "trufor_map_max": float(flat_map.max()),
                  "original_image_size": list(original_size),
                  "inference_image_size": list(inference_size),
                  "input_resized_for_memory": inference_size != original_size,
                  "max_inference_side": MAX_INFERENCE_SIDE,
                  "map_path": str(map_path.resolve()), "raw_output_path": str(raw.resolve()),
                  "device": "cuda", "peak_vram_mb": round(torch.cuda.max_memory_allocated()/1048576, 2)}
        result.update(save_heatmaps(a.image, localization_map, out_dir))
    except Exception as exc:
        result = {"model": "trufor", "success": False,
                  "error_code": "GPU_FAIL" if "GPU_FAIL" in str(exc) else "TRUFOR_INFERENCE_FAILED",
                  "error": str(exc)}
        traceback.print_exc()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

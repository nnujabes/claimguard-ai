"""Real inference jobs for the browser component (no demo/fallback scores)."""
from __future__ import annotations

import base64
import copy
import io
import json
import threading
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from backend.decision import decide
from backend.fusion import fuse
from backend.model_gateway import MODEL_ORDER, ROOT, run_model

CONFIG = json.loads((ROOT / "config/runtime.json").read_text())
GPU_LOCK = threading.Lock()
MAX_FILES = int(CONFIG.get("max_files", 20))
MAX_BYTES = CONFIG["max_upload_mb"] * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024


def image_data_url(path: Path, *, preview=False) -> str:
    if preview:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=88)
        payload, mime = buffer.getvalue(), "image/jpeg"
    else:
        # Bound the browser payload; full-resolution artifacts stay on disk.
        with Image.open(path) as image:
            image.thumbnail((1600, 1600), Image.Resampling.NEAREST)
            buffer = io.BytesIO()
            image.save(buffer, "PNG")
        payload, mime = buffer.getvalue(), "image/png"
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


def prepare_uploads(files: list[dict], directory: Path) -> list[dict]:
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError(f"이미지를 1~{MAX_FILES}장 선택해 주세요.")
    directory.mkdir(parents=True, exist_ok=True)
    prepared, total = [], 0
    for index, item in enumerate(files):
        encoded = item.get("data", "")
        if not isinstance(encoded, str) or len(encoded) > (MAX_BYTES + 2) // 3 * 4:
            raise ValueError("이미지당 최대 20MB까지 업로드할 수 있습니다.")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("이미지 업로드 데이터가 올바르지 않습니다.") from exc
        total += len(payload)
        if not payload or len(payload) > MAX_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError("이미지당 20MB, 전체 100MB까지 업로드할 수 있습니다.")
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("JPEG 또는 PNG 이미지만 지원합니다.")
            image.verify()
        # Normalize orientation once so model maps and the displayed image align.
        path = directory / f"{index:03d}.png"
        with Image.open(io.BytesIO(payload)) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(path)
        prepared.append({"name": Path(str(item.get("name", "image"))).name,
                         "path": path})
    return prepared


def analyze_image(path: Path, name: str, on_progress=None, cancelled=None,
                  threshold: float | None = None) -> dict:
    threshold = CONFIG["roi_threshold"] if threshold is None else float(threshold)
    models = {}
    for model in MODEL_ORDER:
        if cancelled and cancelled.is_set():
            raise InterruptedError("분석 요청이 취소되었습니다.")
        if on_progress:
            on_progress(model, copy.deepcopy(models))
        try:
            result = run_model(model, path, timeout=CONFIG["timeout_seconds"],
                               env_overrides={"CG_STRICT_MODE": "1"})
            if result.get("meta", {}).get("is_official_model") is not True:
                raise RuntimeError(result.get("meta", {}).get("integration_reason")
                                   or result.get("meta", {}).get("reason")
                                   or "공식 모델 추론 결과가 아닙니다.")
            models[model] = {**result, "status": "complete"}
        except Exception as exc:
            models[model] = {"model": model, "status": "error", "score": None,
                             "error": str(exc)[-2000:]}
    complete = all(result["status"] == "complete" for result in models.values())
    scores = {key: result["score"] for key, result in models.items()}
    fusion = fuse(scores, CONFIG["weights"]) if complete else None
    decision = decide(fusion.score, threshold) if complete else None
    artifacts = {}
    fused = models["fused"]
    if fused["status"] == "complete":
        for key in ("heatmap_overlay_path", "mask_path"):
            artifact = fused.get("artifacts", {}).get(key)
            if artifact and Path(artifact).is_file():
                artifacts["heatmap" if key == "heatmap_overlay_path" else "mask"] = image_data_url(Path(artifact))
    return {"name": name, "url": image_data_url(path, preview=True), "models": models,
            "scores": scores, "score": fusion.score if fusion else None,
            "complete": complete, "threshold": threshold,
            "route": decision.route if decision else "INCOMPLETE",
            "dominant_model": fusion.dominant_model if fusion else None,
            "artifacts": artifacts, "part": "업로드 이미지",
            "type": ("추가 검토 필요" if decision.review else "뚜렷한 이상 없음") if decision else "모델 실행 실패"}


class AnalysisJob:
    def __init__(self, request: dict):
        self.request = request
        self.directory = ROOT / "data/runtime/ui" / uuid.uuid4().hex
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.state = {"id": request["id"], "status": "queued", "results": [],
                      "image_index": 0, "total": len(request.get("files", [])),
                      "models": {}, "model": None}

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def start(self):
        threading.Thread(target=self.run, daemon=True, name="claim-model-analysis").start()

    def run(self):
        try:
            threshold = float(self.request.get("threshold", CONFIG["roi_threshold"]))
            if not 0 <= threshold <= 1:
                raise ValueError("ROI 임계값은 0%~100% 범위여야 합니다.")
            files = prepare_uploads(self.request["files"], self.directory)
            self.request = {"id": self.request["id"]}  # Release uploaded base64 payloads.
            with GPU_LOCK:
                results = []
                for index, item in enumerate(files):
                    if self.cancelled.is_set():
                        raise InterruptedError("분석 요청이 취소되었습니다.")
                    self.update(status="running", image_index=index, name=item["name"], models={})
                    result = analyze_image(item["path"], item["name"],
                                           lambda model, models: self.update(model=model, models=models),
                                           self.cancelled, threshold)
                    results.append(result)
                    # Keep a machine-readable record including actual model provenance.
                    (self.directory / "results.json").write_text(
                        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                    self.update(results=list(results), models=result["models"])
                self.update(status="cancelled" if self.cancelled.is_set() else "complete", model=None)
        except InterruptedError:
            self.update(status="cancelled")
        except Exception as exc:
            self.update(status="error", error=str(exc)[-2000:])

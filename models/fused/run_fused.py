from __future__ import annotations
import argparse, json, os, subprocess, sys, time, uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT=ROOT/"weights"/"fused"/"finetuned"/"latest.pth"
CHECKPOINT=Path(os.getenv("CG_FUSED_CHECKPOINT",str(DEFAULT_CHECKPOINT)))
ADAPTER=ROOT/"models"/"fused"/"adapter"/"infer.py"
VENDOR_ENTRY=ROOT/"vendor"/"fused"/"eval"/"opensdi.py"

def format_result(raw:dict,started:float,checkpoint:Path,persistent:bool=False)->dict:
    if not raw.get("success"): raise RuntimeError(raw.get("error","FUSED inference failed"))
    score=float(raw["score"])
    return {"model":"fused","score":score,"risk_label":"HIGH" if score>=.7 else "MEDIUM" if score>=.4 else "LOW",
        "artifacts":{key:raw.get(key) for key in ("mask_path","overlay_path","heatmap_overlay_path","map_path")},
        "mask_path":raw.get("overlay_path") or raw.get("mask_path"),"elapsed_sec":round(time.perf_counter()-started,6),
        "meta":{"checkpoint":str(checkpoint),"checkpoint_present":True,"input_size":[512,512],
        "backend":"official_fused_opensdi","is_official_model":True,"persistent_server":persistent,
        "inference_sec":raw.get("inference_sec"),"peak_vram_mb":raw.get("peak_vram_mb"),
        "checkpoint_load_audit":raw.get("checkpoint_load_audit")}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--image");ap.add_argument("--serve",action="store_true");ap.add_argument("--checkpoint");a=ap.parse_args()
    checkpoint=Path(a.checkpoint or CHECKPOINT)
    if a.serve:
        os.execv(sys.executable,[sys.executable,str(ADAPTER),"--serve","--checkpoint",str(checkpoint)])
    if not a.image: ap.error("--image is required without --serve")
    start=time.perf_counter()
    if not checkpoint.exists(): raise SystemExit(f"FUSED checkpoint missing: {checkpoint}")
    if not VENDOR_ENTRY.exists(): raise SystemExit(f"FUSED official repository missing: {VENDOR_ENTRY.parent.parent}")
    temp=ROOT/"data"/"runtime"/"fused"/uuid.uuid4().hex;temp.mkdir(parents=True,exist_ok=True);result_path=temp/"result.json";out=temp/"out"
    proc=subprocess.run([sys.executable,str(ADAPTER),"--image",a.image,"--output-dir",str(out),"--result-json",str(result_path),"--checkpoint",str(checkpoint)],capture_output=True,text=True)
    if not result_path.exists(): raise SystemExit(proc.stderr.strip() or "FUSED adapter returned no result")
    raw=json.loads(result_path.read_text("utf-8"))
    try: result=format_result(raw,start,checkpoint)
    except RuntimeError as exc: raise SystemExit(str(exc))
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":main()

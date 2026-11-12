from __future__ import annotations
import argparse, json, os, subprocess, sys, time, uuid
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops, ImageFilter

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=Path(os.getenv("CG_TRUFOR_CHECKPOINT",str(ROOT/"weights"/"trufor"/"finetuned"/"best.pth.tar")))
ADAPTER=ROOT/"models"/"trufor"/"adapter"/"infer.py"
VENDOR_ENTRY=ROOT/"vendor"/"trufor"/"TruFor_train_test"/"test.py"

def risk(score): return "HIGH" if score>=.7 else "MEDIUM" if score>=.4 else "LOW"

def official(path:Path):
    if not CHECKPOINT.exists(): raise FileNotFoundError(f"checkpoint missing: {CHECKPOINT}")
    if not VENDOR_ENTRY.exists(): raise FileNotFoundError(f"official TruFor repository missing: {VENDOR_ENTRY.parent}")
    temp=ROOT/"data"/"runtime"/"trufor"/uuid.uuid4().hex;temp.mkdir(parents=True,exist_ok=True);result_path=temp/"result.json";out=temp/"out"
    proc=subprocess.run([sys.executable,str(ADAPTER),"--image",str(path),"--output-dir",str(out),"--result-json",str(result_path),"--checkpoint",str(CHECKPOINT)],capture_output=True,text=True)
    if not result_path.exists(): raise RuntimeError(proc.stderr.strip() or "TruFor adapter returned no result")
    raw=json.loads(result_path.read_text("utf-8"))
    if not raw.get("success"): raise RuntimeError(raw.get("error","TruFor inference failed"))
    score=float(raw["score"])
    return score,raw.get("overlay_path") or raw.get("heatmap_overlay_path"),raw

def ela_score(path:Path):
    image=Image.open(path).convert("RGB").resize((512,512));import io
    buf=io.BytesIO();image.save(buf,"JPEG",quality=90);buf.seek(0);recompressed=Image.open(buf).convert("RGB")
    ela=np.asarray(ImageChops.difference(image,recompressed),dtype=np.float32)
    residual=np.asarray(image.filter(ImageFilter.GaussianBlur(1)),dtype=np.float32)-np.asarray(image,dtype=np.float32)
    return float(np.clip((np.percentile(ela,99)/28+np.std(residual)/22)/2,0,1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--image",required=True);a=ap.parse_args();start=time.perf_counter();path=Path(a.image);reason=None
    try: score,mask,raw=official(path);backend="official_trufor";official_model=True
    except Exception as exc:
        if os.getenv("CG_STRICT_MODE")=="1": raise SystemExit(str(exc))
        reason=str(exc);score=ela_score(path);mask=None;raw={};backend="ela_noise_fallback";official_model=False
    result={"model":"trufor","score":score,"risk_label":risk(score),"mask_path":mask,"elapsed_sec":round(time.perf_counter()-start,6),"meta":{"checkpoint":str(CHECKPOINT),"checkpoint_present":CHECKPOINT.exists(),"input_size":raw.get("inference_image_size",[512,512]),"backend":backend,"is_official_model":official_model,"integration_reason":reason}}
    result["artifacts"]={key:raw.get(key) for key in ("heatmap_overlay_path","map_path")}
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":main()

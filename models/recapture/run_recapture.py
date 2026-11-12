from __future__ import annotations
import argparse, json, os, subprocess, sys, time, uuid
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
ADAPTER=ROOT/"models"/"recapture"/"adapter"/"infer.py"
CHECKPOINTS=ROOT/"weights"/"recapture"
BACKBONE_DIR=ROOT/"downloads"/"recapture_dinov2"/"dinov2-with-registers-base"

def fft_score(path:Path):
    arr=np.asarray(Image.open(path).convert("L").resize((512,512)),dtype=np.float32);arr-=arr.mean();spec=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(arr))));h,w=spec.shape;y,x=np.ogrid[:h,:w];spec[(x-w/2)**2+(y-h/2)**2<(min(h,w)*.06)**2]=0
    return float(np.clip((np.percentile(spec,99.8)-np.median(spec[spec>0]))/8,0,1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--image",required=True);a=ap.parse_args();start=time.perf_counter();reason=None
    ready=(CHECKPOINTS/"best_screen_detector_backbone.pt").exists() and (CHECKPOINTS/"best_screen_detector_mlp.pt").exists() and (BACKBONE_DIR/"config.json").exists() and (BACKBONE_DIR/"model.safetensors").exists()
    if ready:
        temp=ROOT/"data"/"runtime"/"recapture"/uuid.uuid4().hex;temp.mkdir(parents=True,exist_ok=True);result_path=temp/"result.json";out=temp/"out";proc=subprocess.run([sys.executable,str(ADAPTER),"--image",a.image,"--output-dir",str(out),"--result-json",str(result_path)],capture_output=True,text=True)
        raw=json.loads(result_path.read_text("utf-8")) if result_path.exists() else {"success":False,"error":proc.stderr.strip()}
        if raw.get("success"): score=float(raw["score"]);backend="official_dinov2_moire";official=True
        else:
            reason=raw.get("error","recapture adapter failed")
            if os.getenv("CG_STRICT_MODE")=="1": raise SystemExit(reason)
            score=fft_score(Path(a.image));backend="fft_periodicity_fallback";official=False
    else:
        reason=f"Recapture checkpoints or DINOv2 backbone missing: {CHECKPOINTS}, {BACKBONE_DIR}"
        if os.getenv("CG_STRICT_MODE")=="1": raise SystemExit(reason)
        score=fft_score(Path(a.image));backend="fft_periodicity_fallback";official=False
    result={"model":"recapture","score":score,"risk_label":"HIGH" if score>=.7 else "MEDIUM" if score>=.4 else "LOW","mask_path":None,"elapsed_sec":round(time.perf_counter()-start,6),"meta":{"checkpoint":str(CHECKPOINTS),"checkpoint_present":(CHECKPOINTS/"best_screen_detector_backbone.pt").exists() and (CHECKPOINTS/"best_screen_detector_mlp.pt").exists(),"input_size":[224,224],"backend":backend,"is_official_model":official,"integration_reason":reason}}
    print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":main()

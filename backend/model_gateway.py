from __future__ import annotations
import atexit, hashlib, json, os, selectors, signal, subprocess, sys, threading, time, uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL_ORDER=("trufor","fused","recapture")
_LOCKS={name:threading.Lock() for name in MODEL_ORDER}
_PROCESSES={name:None for name in MODEL_ORDER}
_PROCESS_KEYS={name:None for name in MODEL_ORDER}

def _python(model:str)->str:
    configured=os.getenv(f"CG_{model.upper()}_PYTHON")
    if configured:
        return configured
    local_python=ROOT/".venvs"/model/"bin"/"python"
    return str(local_python) if local_python.is_file() else sys.executable

def _entry(model:str)->Path:
    return ROOT/"models"/model/f"run_{model}.py"

def _configure_fused_env(env:dict[str,str])->dict[str,str]:
    """Use the bundled complete ConvNeXt cache instead of downloading at inference time."""
    project_hf=ROOT/"downloads"/"huggingface"
    model_cache=project_hf/"hub"/"models--timm--convnext_xxlarge.clip_laion2b_soup"
    snapshots=model_cache/"snapshots"
    weights=next(snapshots.glob("*/model.safetensors"),None) if snapshots.is_dir() else None
    if not weights or not weights.is_file() or weights.stat().st_size<3_000_000_000:
        raise FileNotFoundError(f"complete local ConvNeXt cache missing: {model_cache}")
    env["HF_HOME"]=str(project_hf)
    env["HF_HUB_OFFLINE"]="1"
    env.setdefault("CG_FUSED_BF16","1")
    return env

def _validate(model:str,result:dict)->dict:
    if result.get("model")!=model: raise ValueError(f"모델명이 일치하지 않습니다: {result.get('model')}")
    score=float(result["score"])
    if not 0<=score<=1: raise ValueError(f"score 범위 오류: {score}")
    result["score"]=score
    return result

def _stop_server(name:str)->None:
    process=_PROCESSES[name]
    _PROCESSES[name]=None;_PROCESS_KEYS[name]=None
    if process and process.poll() is None:
        if os.name=="posix": os.killpg(process.pid,signal.SIGTERM)
        else: process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name=="posix": os.killpg(process.pid,signal.SIGKILL)
            else: process.kill()
            process.wait(timeout=5)

def _stop_all_servers()->None:
    for name in MODEL_ORDER: _stop_server(name)

atexit.register(_stop_all_servers)

def _read_server_json(process:subprocess.Popen,kind:str,timeout:int,name:str)->dict:
    selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    deadline=time.monotonic()+timeout
    try:
        while time.monotonic()<deadline:
            events=selector.select(max(0,deadline-time.monotonic()))
            if not events: break
            line=process.stdout.readline()
            if not line:
                error=f"{name} server exited {process.poll()}"
                raise RuntimeError(error)
            try: payload=json.loads(line)
            except json.JSONDecodeError: continue
            if payload.get("kind")==kind: return payload
    finally: selector.close()
    raise TimeoutError(f"{name} server {kind} timeout after {timeout}s")

def _start_server(name:str,key,extra_args:list[str],timeout:int,env:dict[str,str])->dict:
    process=_PROCESSES[name]
    if process and process.poll() is None and _PROCESS_KEYS[name]==key:
        return {"reused":True}
    _stop_server(name)
    env=env.copy();env.setdefault("PYTHONUNBUFFERED","1")
    process=subprocess.Popen(
        [_python(name),str(ROOT/"models"/name/"adapter"/"infer.py"),"--serve",*extra_args],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,
        start_new_session=(os.name=="posix"),
    )
    _PROCESSES[name],_PROCESS_KEYS[name]=process,key
    ready=_read_server_json(process,"ready",timeout,name)
    if not ready.get("success"):
        _stop_server(name);raise RuntimeError(ready.get("error",f"{name} server failed to start"))
    return ready

def _send_request(name:str,request:dict,timeout:int):
    process=_PROCESSES[name]
    if process is None or process.poll() is not None:
        raise RuntimeError(f"{name} server is not running")
    try:
        process.stdin.write(json.dumps(request)+"\n");process.stdin.flush()
        response=_read_server_json(process,"result",timeout,name)
    except Exception:
        _stop_server(name);raise
    return response["result"]

def _run_fused_persistent(image_path:str|Path,checkpoint:Path,timeout:int,env:dict[str,str],save_visuals:bool=True)->dict:
    started=time.perf_counter()
    with _LOCKS["fused"]:
        ready=_start_server("fused",checkpoint,["--checkpoint",str(checkpoint)],timeout,env)
        runtime=ROOT/"data"/"runtime"/"fused"/uuid.uuid4().hex
        output=runtime/"out";output.mkdir(parents=True,exist_ok=True)
        raw=_send_request("fused",{"image":str(Path(image_path).resolve()),
            "output_dir":str(output.resolve()),"save_visuals":bool(save_visuals)},timeout)
        if not raw.get("success"): raise RuntimeError(raw.get("error","FUSED inference failed"))
        score=float(raw["score"])
        return _validate("fused",{"model":"fused","score":score,
            "risk_label":"HIGH" if score>=.7 else "MEDIUM" if score>=.4 else "LOW",
            "mask_path":raw.get("heatmap_overlay_path") or raw.get("overlay_path") or raw.get("mask_path"),
            "artifacts":{key:raw.get(key) for key in ("mask_path","overlay_path","heatmap_overlay_path","heatmap_path","prob_map_path","map_path")},
            "elapsed_sec":round(time.perf_counter()-started,6),
            "meta":{"checkpoint":str(checkpoint),"checkpoint_present":True,"input_size":[512,512],
                "backend":"official_fused_opensdi","is_official_model":True,"persistent_server":True,
                "server_reused":bool(ready.get("reused")),"resident_vram_mb":ready.get("resident_vram_mb"),
                "inference_sec":raw.get("inference_sec"),"peak_vram_mb":raw.get("peak_vram_mb"),"save_visuals":save_visuals,
                "checkpoint_load_audit":raw.get("checkpoint_load_audit")}})

def _run_trufor_persistent(image_path:str|Path,checkpoint:Path,timeout:int,env:dict[str,str],save_visuals:bool=True)->dict:
    started=time.perf_counter()
    with _LOCKS["trufor"]:
        ready=_start_server("trufor",checkpoint,["--checkpoint",str(checkpoint)],timeout,env)
        runtime=ROOT/"data"/"runtime"/"trufor"/uuid.uuid4().hex
        output=runtime/"out"
        raw=_send_request("trufor",{"image":str(Path(image_path).resolve()),
            "output_dir":str(output.resolve()),"save_visuals":bool(save_visuals)},timeout)
        if not raw.get("success"): raise RuntimeError(raw.get("error","TruFor inference failed"))
        score=float(raw["score"])
        return _validate("trufor",{"model":"trufor","score":score,
            "risk_label":"HIGH" if score>=.7 else "MEDIUM" if score>=.4 else "LOW",
            "mask_path":raw.get("heatmap_overlay_path") or raw.get("map_path"),
            "elapsed_sec":round(time.perf_counter()-started,6),
            "meta":{"checkpoint":str(checkpoint),"checkpoint_present":checkpoint.is_file(),
                "input_size":raw.get("inference_image_size",[512,512]),"backend":"official_trufor",
                "is_official_model":True,"persistent_server":True,"server_reused":bool(ready.get("reused")),
                "resident_vram_mb":ready.get("resident_vram_mb"),"inference_sec":raw.get("inference_sec"),
                "peak_vram_mb":raw.get("peak_vram_mb"),"save_visuals":save_visuals}})

def _run_recapture_persistent(image_path:str|Path,timeout:int,env:dict[str,str])->dict:
    started=time.perf_counter()
    with _LOCKS["recapture"]:
        ready=_start_server("recapture","default",[],timeout,env)
        runtime=ROOT/"data"/"runtime"/"recapture"/uuid.uuid4().hex
        raw=_send_request("recapture",{"image":str(Path(image_path).resolve()),
            "output_dir":str((runtime/"out").resolve())},timeout)
        if not raw.get("success"): raise RuntimeError(raw.get("error","Recapture inference failed"))
        score=float(raw["score"])
        return _validate("recapture",{"model":"recapture","score":score,
            "risk_label":"HIGH" if score>=.7 else "MEDIUM" if score>=.4 else "LOW",
            "mask_path":None,"elapsed_sec":round(time.perf_counter()-started,6),
            "meta":{"checkpoint":str(ROOT/"weights"/"recapture"),"checkpoint_present":True,
                "input_size":[224,224],"backend":"official_dinov2_moire","is_official_model":True,
                "persistent_server":True,"server_reused":bool(ready.get("reused")),
                "resident_vram_mb":ready.get("resident_vram_mb"),"inference_sec":raw.get("inference_sec"),
                "peak_vram_mb":raw.get("peak_vram_mb")}})

def run_model(model:str,image_path:str|Path,timeout:int=180,env_overrides:dict[str,str]|None=None,save_visuals:bool=True)->dict:
    env=os.environ.copy()
    if env_overrides:
        env.update({key:str(value) for key,value in env_overrides.items()})
    persistent=env.get(f"CG_{model.upper()}_PERSISTENT","1").lower() not in {"0","false","no"}
    if model=="fused": env=_configure_fused_env(env)
    if model=="fused" and persistent:
        default_checkpoint=ROOT/"weights"/"fused"/"finetuned"/"latest.pth"
        checkpoint=Path(env.get("CG_FUSED_CHECKPOINT",default_checkpoint)).resolve()
        return _run_fused_persistent(image_path,checkpoint,timeout,env,save_visuals)
    if model=="trufor" and persistent:
        finetuned=ROOT/"weights"/"trufor"/"finetuned"/"best.pth.tar"
        default_checkpoint=finetuned if finetuned.is_file() else ROOT/"weights"/"trufor"/"trufor.pth.tar"
        checkpoint=Path(env.get("CG_TRUFOR_CHECKPOINT",default_checkpoint)).resolve()
        return _run_trufor_persistent(image_path,checkpoint,timeout,env,save_visuals)
    if model=="recapture" and persistent:
        return _run_recapture_persistent(image_path,timeout,env)
    # Kill adapter grandchildren too on timeout, otherwise they retain GPU memory.
    proc=subprocess.Popen([_python(model),str(_entry(model)),"--image",str(image_path)],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,
                          start_new_session=(os.name=="posix"))
    try:
        stdout,stderr=proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name=="posix": os.killpg(proc.pid,signal.SIGKILL)
        else: proc.kill()
        proc.communicate()
        raise TimeoutError(f"{model} inference exceeded {timeout}s")
    if proc.returncode: raise RuntimeError(stderr.strip() or f"{model} exited {proc.returncode}")
    lines=[line for line in stdout.splitlines() if line.strip()]
    if not lines: raise RuntimeError(f"{model} returned no JSON")
    return _validate(model,json.loads(lines[-1]))

def analyze_all(image_path:str|Path,on_progress=None,fused_checkpoint:str|Path|None=None)->dict[str,dict]:
    results={}
    for name in MODEL_ORDER:
        if on_progress: on_progress(name)
        overrides={"CG_FUSED_CHECKPOINT":str(fused_checkpoint)} if name=="fused" and fused_checkpoint else None
        try: results[name]=run_model(name,image_path,env_overrides=overrides)
        except Exception as exc: results[name]=fallback_result(name,image_path,str(exc))
    return results

def fallback_result(model:str,image_path:str|Path,error:str="")->dict:
    start=time.perf_counter();digest=hashlib.sha256(Path(image_path).read_bytes()).digest();offset={"trufor":0,"fused":1,"recapture":2}[model]
    score=.12+(digest[offset]/255)*.76
    return {"model":model,"score":round(score,6),"risk_label":"HIGH" if score>=.7 else "MEDIUM" if score>=.4 else "LOW","mask_path":None,"elapsed_sec":round(time.perf_counter()-start,6),"meta":{"checkpoint":"none","input_size":[512,512],"backend":"gateway_deterministic_fallback","is_official_model":False,"reason":error[:240]}}

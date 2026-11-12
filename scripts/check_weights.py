from pathlib import Path
import hashlib, json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/"weights"/"manifest.json").read_text("utf-8"))
fused_cache_root=ROOT/"downloads/huggingface/hub/models--timm--convnext_xxlarge.clip_laion2b_soup/snapshots"
fused_backbone=next(fused_cache_root.glob("*/model.safetensors"),fused_cache_root/"MISSING") if fused_cache_root.is_dir() else fused_cache_root/"MISSING"
requirements={
 "trufor":[ROOT/"vendor/trufor/TruFor_train_test/test.py"],
 "fused":[ROOT/"vendor/fused/eval/opensdi.py",fused_backbone],
 "recapture":[ROOT/"downloads/recapture_dinov2/dinov2-with-registers-base/config.json",ROOT/"downloads/recapture_dinov2/dinov2-with-registers-base/model.safetensors"],
}
print("ClaimGuard model readiness")
full=True
for name,item in manifest["models"].items():
    rels=[item["path"]] if "path" in item else item["paths"]
    paths=[ROOT/p for p in rels];weights_ok=all(p.is_file() and p.stat().st_size>(0 if p.suffix==".json" else 1024) for p in paths)
    deps_ok=all(p.exists() and (p!=fused_backbone or p.stat().st_size>3_000_000_000) for p in requirements[name]);ready=weights_ok and deps_ok;full &= ready
    print(f"- {name:9} weights={'READY' if weights_ok else 'MISSING'}  runtime={'READY' if ready else 'INCOMPLETE'}")
    if not deps_ok: print("  missing:",", ".join(str(p.relative_to(ROOT)) for p in requirements[name] if not p.exists()))
print("\nSource asset mode:","COMPLETE" if full else "MIXED / FALLBACK")
print("CUDA and each model's Python dependencies must also be available for official inference.")

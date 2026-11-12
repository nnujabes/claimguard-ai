from __future__ import annotations
import json
from pathlib import Path

class ResultRepository:
    def __init__(self,root="data/results"): self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def save(self,claim_id,result):
        path=self.root/f"{claim_id}.json";data=json.loads(path.read_text("utf-8")) if path.exists() else {"claim_id":claim_id,"analyses":[]};data["analyses"].append(result)
        tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8");tmp.replace(path);return path
    def save_many(self,claim_id,results):
        path=self.root/f"{claim_id}.json";data=json.loads(path.read_text("utf-8")) if path.exists() else {"claim_id":claim_id,"analyses":[]};data["analyses"].extend(results)
        tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8");tmp.replace(path);return path
    def list_claims(self): return [json.loads(p.read_text("utf-8")) for p in sorted(self.root.glob("*.json"),reverse=True)]
    def get(self,claim_id): return json.loads((self.root/f"{claim_id}.json").read_text("utf-8"))

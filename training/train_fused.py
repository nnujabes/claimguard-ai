from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from model_adapter import build_model

PARTS = {"f_bmp":"앞범퍼","f_fnd":"앞펜더","light":"전조등","r_bmp":"뒷범퍼","r_fnd":"뒷펜더","fd":"앞도어","bd":"뒷도어","sm":"사이드미러","bnt":"본넷","trk":"트렁크"}

def discover(root: Path):
    records=[]
    for edited in sorted(root.glob("*_2_edi.*")):
        stem=edited.stem.replace("_2_edi","")
        originals=list(root.glob(stem+"_1_org.*")); masks=list(root.glob(stem+"_3_msk.*"))
        if not originals or not masks: continue
        part=next((v for k,v in PARTS.items() if stem.startswith(k)),"기타")
        records += [{"image":str(originals[0]),"mask":None,"label":0,"group":stem,"part":part},{"image":str(edited),"mask":str(masks[0]),"label":1,"group":stem,"part":part}]
    if not records: raise FileNotFoundError("원본/조작본/마스크 3종 세트를 찾지 못했습니다.")
    return records

def split_groups(records, seed=42):
    groups=sorted({r["group"] for r in records}); random.Random(seed).shuffle(groups)
    n=len(groups); cuts=(int(n*.7),int(n*.85)); buckets={g:("train" if i<cuts[0] else "val" if i<cuts[1] else "test") for i,g in enumerate(groups)}
    return {s:[r for r in records if buckets[r["group"]]==s] for s in ("train","val","test")}

class ClaimDataset(Dataset):
    def __init__(self, rows, size=512, train=False): self.rows,self.size,self.train=rows,size,train
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; image=Image.open(r["image"]).convert("RGB").resize((self.size,self.size),Image.Resampling.BILINEAR)
        mask=Image.open(r["mask"]).convert("L").resize((self.size,self.size),Image.Resampling.NEAREST) if r["mask"] else Image.new("L",(self.size,self.size))
        x=torch.from_numpy(np.asarray(image,dtype=np.float32).transpose(2,0,1)/255.); y=torch.from_numpy((np.asarray(mask)>127).astype(np.float32))[None]
        if self.train and random.random()<.5: x,y=torch.flip(x,[2]),torch.flip(y,[2])
        return x,torch.tensor(r["label"],dtype=torch.float32),y

def dice_loss(logits,target):
    p=logits.sigmoid(); return 1-(2*(p*target).sum((1,2,3))+1)/((p+target).sum((1,2,3))+1)

def evaluate(model,loader,device):
    model.eval(); correct=total=iou_sum=0
    with torch.no_grad():
        for x,y,m in loader:
            out=model(x.to(device)); pred=(out["logits"].sigmoid()>=.5).cpu(); correct+=(pred==y.bool()).sum().item();total+=y.numel()
            pm=out["mask_logits"].sigmoid()>=.5; tm=m.to(device)>0; inter=(pm&tm).sum((1,2,3)); union=(pm|tm).sum((1,2,3)); iou_sum+=((inter+1)/(union+1)).sum().item()
    return {"accuracy":correct/max(total,1),"mean_iou":iou_sum/max(total,1)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,default=Path("checkpoints"));p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--image-size",type=int,default=512);p.add_argument("--lr",type=float,default=2e-4);p.add_argument("--checkpoint");a=p.parse_args()
    torch.manual_seed(42); splits=split_groups(discover(a.data_dir));a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/"split_manifest.json").write_text(json.dumps(splits,ensure_ascii=False,indent=2),encoding="utf-8")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=build_model(a.checkpoint).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);bce=nn.BCEWithLogitsLoss();best=-1
    loaders={s:DataLoader(ClaimDataset(rows,a.image_size,s=="train"),batch_size=a.batch_size,shuffle=s=="train",num_workers=0) for s,rows in splits.items()}
    for epoch in range(1,a.epochs+1):
        model.train();running=0
        for x,y,m in loaders["train"]:
            x,y,m=x.to(device),y.to(device),m.to(device);out=model(x);loss=bce(out["logits"],y)+.7*bce(out["mask_logits"],m)+.3*dice_loss(out["mask_logits"],m).mean();opt.zero_grad();loss.backward();opt.step();running+=loss.item()
        metrics=evaluate(model,loaders["val"],device);score=metrics["accuracy"]+metrics["mean_iou"]
        print(json.dumps({"epoch":epoch,"loss":running/max(len(loaders["train"]),1),**metrics},ensure_ascii=False))
        if score>best: best=score;torch.save({"model":model.state_dict(),"epoch":epoch,"metrics":metrics},a.output_dir/"fused_best.pt")
    print(json.dumps({"test":evaluate(model,loaders["test"],device)},ensure_ascii=False))
if __name__=="__main__": main()

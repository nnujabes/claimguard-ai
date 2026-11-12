from pathlib import Path
import numpy as np
from PIL import Image

def save_heatmaps(image_path,localization_map,output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True);base=Image.open(image_path).convert("RGB");w,h=base.size
    values=np.asarray(localization_map,dtype=np.float32);values=np.nan_to_num(values);values=(values-values.min())/(values.max()-values.min()+1e-8)
    heat=np.zeros((*values.shape,4),dtype=np.uint8);heat[...,0]=(255*values).astype(np.uint8);heat[...,1]=(210*np.clip(1-np.abs(values-.55)*2,0,1)).astype(np.uint8);heat[...,3]=(190*values).astype(np.uint8)
    resampling=getattr(Image,"Resampling",Image)
    heat_img=Image.fromarray(heat,"RGBA").resize((w,h),resampling.BILINEAR);heat_path=output_dir/"heatmap.png";overlay_path=output_dir/"heatmap_overlay.png";heat_img.save(heat_path);Image.alpha_composite(base.convert("RGBA"),heat_img).convert("RGB").save(overlay_path)
    return {"heatmap_path":str(heat_path.resolve()),"heatmap_overlay_path":str(overlay_path.resolve())}

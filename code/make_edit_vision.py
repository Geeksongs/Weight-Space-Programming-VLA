import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from safetensors.torch import load_file
from apply_edit import save_edit
# usage: make_edit_vision.py NAME MODE   MODE in gray | killR | killG | killB | softgray
name, mode = sys.argv[1], sys.argv[2]
base=load_file("models/smolvla_libero_official/model.safetensors")
k=[k for k in base if k.endswith("vision_model.embeddings.patch_embedding.weight")][0]
W=base[k].float().clone(); print(k, tuple(W.shape))     # (out, 3, p, p)   channel order RGB
if mode=="gray": W=W.mean(1,keepdim=True).expand_as(W).clone()
elif mode=="softgray": W=0.5*W+0.5*W.mean(1,keepdim=True)
elif mode=="killR": W[:,0]=0
elif mode=="killG": W[:,1]=0
elif mode=="killB": W[:,2]=0
save_edit(f"runs/run1/edit_{name}",{k:W},base); print(name,mode)

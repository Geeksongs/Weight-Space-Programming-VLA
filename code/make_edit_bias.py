import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from safetensors.torch import load_file
from apply_edit import save_edit
# usage: make_edit_bias.py NAME sx,sy,sz,srx,sry,srz,sg   unconditional shifts (normalized action units) via action_out_proj.bias
name, sh = sys.argv[1], [float(v) for v in sys.argv[2].split(",")]
base=load_file("models/smolvla_libero_official/model.safetensors"); b=base["model.action_out_proj.bias"].float().clone()
for i,s in enumerate(sh): b[i]+=s
save_edit(f"runs/run1/edit_{name}",{"model.action_out_proj.bias":b},base); print(name,sh)

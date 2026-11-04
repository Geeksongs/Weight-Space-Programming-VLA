import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from safetensors.torch import load_file
from apply_edit import save_edit
name, sites, alpha = sys.argv[1], sys.argv[2].split(","), float(sys.argv[3])
P=torch.load("runs/run1/act2_pert.pt"); G=torch.load("runs/run1/act2_good.pt")
base=load_file("models/smolvla_libero_official/model.safetensors"); edits={}
for s in sites:
    d=P[s].mean(0)-G[s].mean(0); u=d/d.norm()
    k=("model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight" if s=="connector"
       else f"model.vlm_with_expert.vlm.model.text_model.layers.{s[3:]}.mlp.down_proj.weight")
    W=base[k].float(); edits[k]=W-alpha*torch.outer(u,u@W)
man=save_edit(f"runs/run1/edit_{name}",edits,base); print(name,sum(v['changed'] for v in man.values()),"entries in",len(man),"tensors")

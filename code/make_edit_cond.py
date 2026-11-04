import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from safetensors.torch import load_file
from apply_edit import save_edit
name, s_x, s_z = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])    # shifts (normalized action units) applied when the scene looks "perturbed"
R=torch.load("runs/run1/head_in.pt")
G=torch.stack([t.mean(dim=(0,1,2)) for t in R["libero_object"]]); P=torch.stack([t.mean(dim=(0,1,2)) for t in R["libero_object_object"]])
d=P.mean(0)-G.mean(0); u=d/d.norm(); mg,mp=float((G@u).mean()),float((P@u).mean()); D=mp-mg
base=load_file("models/smolvla_libero_official/model.safetensors")
W=base["model.action_out_proj.weight"].float().clone(); b=base["model.action_out_proj.bias"].float().clone()
for dim,s in ((0,s_x),(2,s_z)):
    if s==0: continue
    W[dim]+= (s/D)*u; b[dim]+= -s*mg/D           # shift = s*(u.h-mg)/D : 0 on original scenes, s on perturbed ones
man=save_edit(f"runs/run1/edit_{name}",{"model.action_out_proj.weight":W,"model.action_out_proj.bias":b},base)
print(name,"s_x",s_x,"s_z",s_z,"changed",{k:v['changed'] for k,v in man.items()})

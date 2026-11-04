import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from safetensors.torch import load_file
from apply_edit import save_edit
def main(name, layers, alpha=1.0):
    S=torch.load("runs/run1/act_swap.pt"); B=torch.load("runs/run1/act_base.pt")
    base=load_file("models/smolvla_libero_official/model.safetensors")
    edits={}
    for i in layers:
        d=torch.cat(S[i]).mean(0)-torch.cat(B[i]).mean(0); u=(d/d.norm())
        k=f"model.vlm_with_expert.lm_expert.layers.{i}.mlp.down_proj.weight"
        W=base[k].float()                      # (720, 2048): output dim = 720
        edits[k]=W-alpha*torch.outer(u,u@W)    # remove direction u from what this MLP writes
    man=save_edit(f"runs/run1/edit_{name}",edits,base)
    print(name,{k.split('layers.')[1]:v['changed'] for k,v in man.items()})
if __name__=="__main__":
    main(sys.argv[1],[int(x) for x in sys.argv[2].split(",")],float(sys.argv[3]) if len(sys.argv)>3 else 1.0)

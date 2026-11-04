"""Teacher-free activation read (counts as forward passes). Records mean lm_expert MLP down_proj outputs per chunk."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from libero.libero import benchmark
BUDGET="/media/python/Seagate/research/smovla_astra/runs/run1/budget.json"
suite_name, task, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
eps=[0,1,2,3]
b=json.load(open(BUDGET)); b["used"]+=1; b["log"].append(dict(tag=tag,kind="activation_read",suite=suite_name,task=task,eps=eps)); json.dump(b,open(BUDGET,"w"),indent=1)
suite=benchmark.get_benchmark_dict()[suite_name]()
policy,*procs=ev.load_policy()
layers=policy.model.vlm_with_expert.lm_expert.layers
store={i:[] for i in range(len(layers))}
def mk(i):
    def h(m,inp,out):
        store[i].append(out.detach().float().mean(dim=(0,1)).cpu())   # mean over batch & tokens
    return h
for i,l in enumerate(layers): l.mlp.down_proj.register_forward_hook(mk(i))
n_chunks=[]
for e in eps:
    for i in store: store[i].append(None)   # episode marker
    ev.run_episode(policy,procs,suite,suite_name,task,e,seed=1000+e)
# collapse: average over denoise steps per chunk is implicit; keep per-episode mean over all calls
out={}
for i,v in store.items():
    cur=[];eps_m=[]
    for x in v:
        if x is None:
            if cur: eps_m.append(torch.stack(cur))
            cur=[]
        else: cur.append(x)
    eps_m.append(torch.stack(cur)) if cur else None
    out[i]=eps_m[-4:]
    # eps_m aligned with episodes (marker placed before each; first group empty)
torch.save({i:[t for t in v] for i,v in out.items()}, f"/media/python/Seagate/research/smovla_astra/runs/run1/act_{tag}.pt")
print("saved", tag, {i:[tuple(t.shape) for t in v] for i,v in list(out.items())[:1]})

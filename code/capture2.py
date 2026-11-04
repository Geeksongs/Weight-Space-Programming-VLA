"""Activation read (1 forward pass). Mean activation at connector output and VLM text-layer MLP outputs."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from libero.libero import benchmark
BUD="/media/python/Seagate/research/smovla_astra/runs/run1/budget.json"
sn, tid, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]; eps=[0,1,2,3]
b=json.load(open(BUD)) if os.path.exists(BUD) else dict(used=0,log=[]); b["used"]+=1; b["log"].append(dict(tag=tag,kind="activation_read",suite=sn,task=tid)); json.dump(b,open(BUD,"w"),indent=1)
suite=benchmark.get_benchmark_dict()[sn](); policy,*procs=ev.load_policy()
vlm=policy.model.vlm_with_expert.vlm.model
sites={"connector":vlm.connector.modality_projection.proj}
for i,l in enumerate(vlm.text_model.layers): sites[f"vlm{i}"]=l.mlp.down_proj
store={k:[] for k in sites}
for k,m in sites.items(): m.register_forward_hook(lambda mod,inp,out,k=k: store[k].append(out.detach().float().reshape(-1,out.shape[-1]).mean(0).cpu()))
for e in eps: ev.run_episode(policy,procs,suite,sn,tid,e,seed=1000+e)
torch.save({k:torch.stack(v) for k,v in store.items()}, f"/media/python/Seagate/research/smovla_astra/runs/run1/act2_{tag}.pt")
print("saved",tag,{k:tuple(torch.stack(v).shape) for k,v in list(store.items())[:2]})

"""One read (1 forward pass): connector output (image tokens) at the first inference step, original vs perturbed scene, eps 0-3."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from libero.libero import benchmark
BUD="/media/python/Seagate/research/smovla_astra/runs/run1/budget.json"
b=json.load(open(BUD)); b["used"]+=1; b["log"].append(dict(tag="tok_read",kind="token_read")); json.dump(b,open(BUD,"w"),indent=1)
policy,pre,post,env_pre,env_post=ev.load_policy()
conn=policy.model.vlm_with_expert.vlm.model.connector.modality_projection.proj
got=[]; conn.register_forward_hook(lambda m,i,o: got.append(o.detach().float().cpu()))
res={}
for sn in ("libero_object","libero_object_object"):
    suite=benchmark.get_benchmark_dict()[sn](); res[sn]=[]
    for ep in range(4):
        env=LiberoEnv(task_suite=suite,task_id=0,task_suite_name="libero_object",episode_length=10,obs_type="pixels_agent_pos",episode_index=ep,observation_height=256,observation_width=256)
        obs,_=env.reset(seed=1000+ep); lang=env.task_description; env.close()
        o={"pixels":{k:v[None] for k,v in obs["pixels"].items()},"robot_state":{g:{k:v[None] for k,v in d.items()} for g,d in obs["robot_state"].items()}}
        o=preprocess_observation(o); o["task"]=[lang]; o=pre(env_pre(o)); policy.reset(); got.clear()
        with torch.inference_mode(): policy.predict_action_chunk(o)
        res[sn].append(got[0].clone()); 
print({k:[tuple(t.shape) for t in v][:1] for k,v in res.items()})
torch.save(res,"runs/run1/tok_read.pt")

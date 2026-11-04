"""1 forward-pass read: inputs of action_out_proj (expert final hidden) averaged over action tokens & denoise steps, first chunk only."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from libero.libero import benchmark
BUD="/media/python/Seagate/research/smovla_astra/runs/run1/budget.json"
b=json.load(open(BUD)); b["used"]+=1; b["log"].append(dict(tag="head_in_read",kind="head_input_read")); json.dump(b,open(BUD,"w"),indent=1)
policy,pre,post,env_pre,env_post=ev.load_policy()
got=[]; policy.model.action_out_proj.register_forward_pre_hook(lambda m,i: got.append(i[0].detach().float().cpu()))
res={}
for sn in ("libero_object","libero_object_object"):
    suite=benchmark.get_benchmark_dict()[sn](); res[sn]=[]
    for ep in range(4):
        env=LiberoEnv(task_suite=suite,task_id=0,task_suite_name="libero_object",episode_length=10,obs_type="pixels_agent_pos",episode_index=ep,observation_height=256,observation_width=256)
        obs,_=env.reset(seed=1000+ep); lang=env.task_description; env.close()
        o={"pixels":{k:v[None] for k,v in obs["pixels"].items()},"robot_state":{g:{k:v[None] for k,v in d.items()} for g,d in obs["robot_state"].items()}}
        o=preprocess_observation(o); o["task"]=[lang]; o=pre(env_pre(o)); policy.reset(); got.clear()
        with torch.inference_mode(): policy.predict_action_chunk(o)
        res[sn].append(torch.stack(got))        # (10 denoise steps, 1, 50, 720)
print({k:tuple(v[0].shape) for k,v in res.items()})
torch.save(res,"runs/run1/head_in.pt")

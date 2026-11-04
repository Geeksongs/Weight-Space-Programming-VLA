import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from libero.libero import benchmark
policy,pre,post,env_pre,env_post = ev.load_policy()
def first_obs(sn, tid, ep):
    suite = benchmark.get_benchmark_dict()[sn]()
    env = LiberoEnv(task_suite=suite, task_id=tid, task_suite_name="libero_spatial", episode_length=10,
                    obs_type="pixels_agent_pos", episode_index=ep, observation_height=256, observation_width=256)
    obs,_ = env.reset(seed=1000+ep); lang = env.task_description; env.close(); return obs, lang
def chunk(obs, lang, noise):
    o = {"pixels":{k:v[None] for k,v in obs["pixels"].items()},
         "robot_state":{g:{k:v[None] for k,v in d.items()} for g,d in obs["robot_state"].items()}}
    o = preprocess_observation(o); o["task"]=[lang]; o = pre(env_pre(o))
    policy.reset()
    with torch.inference_mode():
        return policy.predict_action_chunk(o, noise=noise)[0].float().cpu()   # (50, 32)
noise = None
res = {}
for tid in (3, 1):
    ob, lb = first_obs("libero_spatial", tid, 0); os_, ls = first_obs("libero_spatial_swap", tid, 0)
    if noise is None:
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")
    a_b, a_s = chunk(ob, lb, noise), chunk(os_, ls, noise)
    a_b2 = chunk(ob, lb, torch.randn_like(noise))
    d_scene = (a_b - a_s)[:, :3].abs().mean().item(); d_noise = (a_b - a_b2)[:, :3].abs().mean().item()
    print(f"task{tid}: |chunk(orig)-chunk(swap)| xyz={d_scene:.3f}  vs noise-only={d_noise:.3f}  | lang same: {lb==ls}")
    print("  orig first-20-step mean xyz", np.round(a_b[:20,:3].mean(0).numpy(),2), " swap", np.round(a_s[:20,:3].mean(0).numpy(),2))
    print("  orig steps20-50 mean xyz  ", np.round(a_b[20:50,:3].mean(0).numpy(),2), " swap", np.round(a_s[20:50,:3].mean(0).numpy(),2))

"""Evaluate SmolVLA (LIBERO-finetuned) on a LIBERO / LIBERO-PRO suite task.

Usage:
  python eval_policy.py --suite libero_spatial_swap --tasks 0,1 --episodes 10 [--edit EDIT_DIR] [--out out.json]

Runs ONE env per episode (sequential); init state index = episode index (mod #init states).
No gradients anywhere. An optional --edit dir (manifest.json + edit.npz) is applied to the weights.
"""
import os, sys, json, time, argparse
os.environ.setdefault("LIBERO_CONFIG_PATH", "/media/python/Seagate/research/libero/.libero_config")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/media/python/Seagate/research/hf_cache")
sys.path.insert(0, "/media/python/Seagate/research/libero")  # Pro libero first

import numpy as np, torch
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv, TASK_SUITE_MAX_STEPS
from lerobot.envs.configs import LiberoEnv as LiberoCfg
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

MODEL = os.environ.get("MODEL_DIR", "/media/python/Seagate/research/smovla_astra/models/smolvla_libero_official")
_DEFAULT_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def base_suite(name):
    for b in _DEFAULT_STEPS:
        if name.startswith(b) and (b != "libero_10" or name.startswith("libero_10")):
            return b
    return name


def load_policy(edit_dir=None, device="cuda"):
    policy = SmolVLAPolicy.from_pretrained(MODEL)
    if edit_dir:
        from apply_edit import apply_edit_to_policy
        apply_edit_to_policy(policy, edit_dir)
    policy.to(device).eval()
    assert next(policy.parameters()).is_cuda, 'policy not on GPU'
    print('policy device:', next(policy.parameters()).device, torch.cuda.get_device_name(0), flush=True)
    cfg = policy.config
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=MODEL,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    env_cfg = LiberoCfg(task="libero_10")
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    return policy, pre, post, env_pre, env_post


def run_episode(policy, procs, suite, suite_name, task_id, ep_idx, seed, max_steps=None, hook=None, video=None):
    pre, post, env_pre, env_post = procs
    max_steps = max_steps or _DEFAULT_STEPS.get(base_suite(suite_name), 400)
    env = LiberoEnv(task_suite=suite, task_id=task_id, task_suite_name=base_suite(suite_name),
                    episode_length=max_steps, obs_type="pixels_agent_pos", episode_index=ep_idx,
                    observation_height=256, observation_width=256)
    try:
        obs, _ = env.reset(seed=seed)
        policy.reset()
        success, steps, actions = False, 0, []
        eef = []
        frames = []
        for t in range(max_steps):
            o = {"pixels": {k: v[None] for k, v in obs["pixels"].items()},
                 "robot_state": {g: {k: v[None] for k, v in d.items()} for g, d in obs["robot_state"].items()}}
            o = preprocess_observation(o)
            o["task"] = [env.task_description]
            o = env_pre(o)
            o = pre(o)
            with torch.inference_mode():
                a = policy.select_action(o)
            a = post(a)
            a = env_post({"action": a})["action"]
            a_np = a.to("cpu").numpy()[0]
            actions.append(a_np)
            eef.append(np.concatenate([obs['robot_state']['eef']['pos'], obs['robot_state']['gripper']['qpos']]))
            if video is not None and (t % 4 == 0 if video is True else t in video):
                frames.append((obs["pixels"]["image"][::-1, ::-1].copy(), obs["pixels"]["image2"][::-1, ::-1].copy()) if video is not True else obs["pixels"]["image"][::-1, ::-1].copy())
            obs, r, term, trunc, info = env.step(a_np)
            steps = t + 1
            if info.get("is_success", False):
                success = True
                break
            if term or trunc:
                break
        return dict(success=bool(success), steps=steps, task=env.task_description,
                    actions=np.asarray(actions, dtype=np.float32), eef=np.asarray(eef)), frames
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--ep_offset", type=int, default=0)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed_base", type=int, default=1000)
    a = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[a.suite]()
    n = len(suite.tasks)
    tasks = list(range(n)) if a.tasks == "all" else [int(x) for x in a.tasks.split(",")]
    policy, *procs = load_policy(a.edit)
    res = {}
    for tid in tasks:
        succ, steps = [], []
        for e in range(a.ep_offset, a.ep_offset + a.episodes):
            t0 = time.time()
            r, _ = run_episode(policy, procs, suite, a.suite, tid, e, seed=a.seed_base + e)
            succ.append(r["success"]); steps.append(r["steps"])
            print(f"[{a.suite} t{tid} ep{e}] success={r['success']} steps={r['steps']} {time.time()-t0:.1f}s", flush=True)
        res[tid] = dict(task=suite.tasks[tid].language, rate=float(np.mean(succ)), n=len(succ), successes=succ, steps=steps)
        print(f"== {a.suite} task {tid}: {res[tid]['rate']:.2f}  | {res[tid]['task']}", flush=True)
    if a.out:
        json.dump(dict(suite=a.suite, edit=a.edit, results=res), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

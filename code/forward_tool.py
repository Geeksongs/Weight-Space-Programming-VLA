"""Budgeted forward-pass tool for the editor ("Astra"). One call = one 'forward pass' = rollouts of
<= MAX_EPS dev episodes. Dev episode indices are 0..9 only; test episodes (>=100) are never exposed.
Usage: python forward_tool.py --suite libero_spatial_swap --task 3 --eps 0,1,2,3 [--edit DIR] --tag name
"""
import sys, os, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from libero.libero import benchmark
BUDGET_FILE = "/media/python/Seagate/research/smovla_astra/runs/run1/budget.json"
TOTAL, MAX_EPS = 50, 10

ap = argparse.ArgumentParser()
ap.add_argument("--suite"); ap.add_argument("--task", type=int); ap.add_argument("--eps", default="0,1,2,3")
ap.add_argument("--edit", default=None); ap.add_argument("--tag", default="fwd")
a = ap.parse_args()
eps = [int(x) for x in a.eps.split(",")]
assert len(eps) <= MAX_EPS and all(0 <= e < 10 for e in eps), "dev episodes must be 0..9"
os.makedirs(os.path.dirname(BUDGET_FILE), exist_ok=True)
import fcntl
def _budget(delta_entry=None, check=False):
    with open(BUDGET_FILE + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        b = json.load(open(BUDGET_FILE)) if os.path.exists(BUDGET_FILE) else dict(used=0, log=[])
        if check: assert b["used"] < TOTAL, "forward-pass budget exhausted"
        if delta_entry: b["log"].append(delta_entry); b["used"] = len(b["log"])
        json.dump(b, open(BUDGET_FILE, "w"), indent=1); return b
_budget(check=True)    # charged only after the pass succeeds (failed requests do not count, as in the blog)

suite = benchmark.get_benchmark_dict()[a.suite]()
policy, *procs = ev.load_policy(a.edit)
out = dict(); d = f"/media/python/Seagate/research/smovla_astra/runs/run1/fwd/{a.tag}"; os.makedirs(d, exist_ok=True)
for e in eps:
    r, _ = ev.run_episode(policy, procs, suite, a.suite, a.task, e, seed=1000 + e)
    np.savez(f"{d}/ep{e}.npz", eef=r["eef"], actions=r["actions"])
    print(f"ep{e} success={r['success']} steps={r['steps']}")
    out[e] = r["success"]
b = _budget(dict(tag=a.tag, suite=a.suite, task=a.task, eps=eps, edit=a.edit))
print(f"SUCCESS {sum(out.values())}/{len(out)}  | budget used {b['used']}/{TOTAL} | saved {d}")

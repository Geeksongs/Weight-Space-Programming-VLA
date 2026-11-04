import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from libero.libero import benchmark
from PIL import Image
policy,*procs = ev.load_policy()
out = {}
for sn in ("libero_spatial", "libero_spatial_swap"):
    suite = benchmark.get_benchmark_dict()[sn]()
    r, fr = ev.run_episode(policy, procs, suite, sn, 3, 0, seed=1000, video=True)   # frames every 4 steps
    pick = [0, 10, 15, 20] if len(fr) > 20 else list(range(0, len(fr), 5))
    out[sn] = [fr[i] for i in pick if i < len(fr)]
    print(sn, "frames", len(fr), "success", r["success"])
rows = [np.concatenate(v, axis=1) for v in out.values()]
Image.fromarray(np.concatenate(rows, axis=0)).save("runs/run1/frames_t3.png")

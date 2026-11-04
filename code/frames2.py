import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from libero.libero import benchmark
from PIL import Image
sn_list = sys.argv[1].split(","); tid = int(sys.argv[2]); ep = int(sys.argv[3]); outp = sys.argv[4]
policy,*procs = ev.load_policy(); rows=[]
for sn in sn_list:
    suite = benchmark.get_benchmark_dict()[sn]()
    r, fr = ev.run_episode(policy, procs, suite, sn, tid, ep, seed=1000+ep, video=True)
    n=len(fr); pick=[0, n//4, n//2, (3*n)//4, n-1]
    print(sn, "frames", n, "success", r["success"], "steps", r["steps"])
    rows.append(np.concatenate([fr[i] for i in pick], axis=1))
Image.fromarray(np.concatenate(rows, axis=0)).save(outp)

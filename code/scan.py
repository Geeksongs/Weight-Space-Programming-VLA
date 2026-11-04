import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from libero.libero import benchmark
policy,*procs = ev.load_policy()
d = benchmark.get_benchmark_dict()
for sn in sys.argv[1].split(","):
    try: suite = d[sn]()
    except Exception as e: print("SKIP", sn, e); continue
    n = len(suite.tasks)
    for tid in range(0, n, max(1, n // 2)):
        s = []
        for e in range(int(sys.argv[2])):
            r,_ = ev.run_episode(policy, procs, suite, sn, tid, e, seed=1000+e); s.append(r["success"])
        print(f"SCAN {sn} t{tid} {sum(s)}/{len(s)} | {suite.tasks[tid].language}", flush=True)

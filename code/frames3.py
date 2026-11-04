import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import eval_policy as ev
from libero.libero import benchmark
from PIL import Image
policy,*procs = ev.load_policy(); steps=[36,42,46,50,54]
rows=[]
for sn,ep in (("libero_object",0),("libero_object_object",0)):
    suite=benchmark.get_benchmark_dict()[sn](); r,fr=ev.run_episode(policy,procs,suite,sn,0,ep,seed=1000+ep,video=steps)
    print(sn,"success",r["success"],"frames",len(fr))
    rows.append(np.concatenate([f[0] for f in fr],axis=1)); rows.append(np.concatenate([f[1] for f in fr],axis=1))
Image.fromarray(np.concatenate(rows,axis=0)).save("runs/run1/frames_stuck.png")

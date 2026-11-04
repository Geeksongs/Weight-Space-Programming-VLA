import sys, numpy as np, os
for tag in sys.argv[1:]:
    d=f"/media/python/Seagate/research/smovla_astra/runs/run1/fwd/{tag}"; deep=0; carry=0; n=0; rows=[]
    for e in range(10):
        p=f"{d}/ep{e}.npz"
        if not os.path.exists(p): continue
        z=np.load(p); x=z["eef"]; a=z["actions"]; c=np.where(a[:,6]>0)[0]
        if not len(c): continue
        f=int(c[0]); n+=1; w=float(x[min(f+10,len(x)-1),3])
        held=sum(1 for t in range(f+10,len(x)) if x[t,3]>0.012 and a[t,6]>0)
        deep+= (x[f,2]<0.09 and w>0.01); carry+= held>=60
        rows.append((x[f,0],x[f,1],x[f,2],w))
    print(f"{tag:34s} deep-grasp {deep}/{n} carry>=60 {carry}/{n}  mean(x,y,z,w)", np.round(np.mean(rows,0),3).tolist() if rows else None)

import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from safetensors.torch import load_file
from apply_edit import save_edit
name, cols, scale = sys.argv[1], [int(c) for c in sys.argv[2].split(",")], float(sys.argv[3])
base = load_file("models/smolvla_libero_official/model.safetensors")
k = "model.state_proj.weight"; W = base[k].float().clone(); print(k, tuple(W.shape))
for c in cols: W[:, c] *= scale
man = save_edit(f"runs/run1/edit_{name}", {k: W}, base); print(name, man)

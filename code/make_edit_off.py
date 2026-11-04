import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from safetensors.torch import load_file
from apply_edit import save_edit
name, dz, AXIS = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
base = load_file("models/smolvla_libero_official/model.safetensors")
st = load_file("models/smolvla_libero_official/policy_preprocessor_step_5_normalizer_processor.safetensors")
std_z = [v for k,v in st.items() if "state" in k and "std" in k][0][AXIS].item()
W = base["model.state_proj.weight"].float(); b = base["model.state_proj.bias"].float()
man = save_edit(f"runs/run1/edit_{name}", {"model.state_proj.bias": b + W[:,AXIS] * (dz/std_z)}, base)
print(name, "dz", dz, "std_z", round(std_z,4), man)

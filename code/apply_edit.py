"""Apply a weight edit (manifest.json + edit.npz) to a SmolVLA policy. No training involved.
edit.npz holds, per edited tensor `name`: name+'::idx' (flat int64 indices) and name+'::val' (new values).
manifest.json records shapes, #changed entries and sha256 of the npz arrays.
"""
import json, hashlib, os, numpy as np, torch

def save_edit(out_dir, edits, base_state):
    """edits: {tensor_name: new_tensor(torch, same shape as base)}. Saves only changed entries."""
    os.makedirs(out_dir, exist_ok=True)
    arrs, man = {}, {}
    for n, new in edits.items():
        base = base_state[n].float().cpu().flatten()
        newf = new.float().cpu().flatten()
        assert base.shape == newf.shape, f"shape mismatch {n}"
        idx = torch.nonzero(newf != base).flatten().numpy().astype(np.int64)
        if len(idx) == 0:
            continue
        # refuse "multiply whole tensor by a constant"
        ratio = (newf[idx] / base[idx].clamp(min=1e-12)) if len(idx) == len(base) else None
        if ratio is not None and torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-4):
            raise ValueError(f"{n}: whole-tensor constant scaling is not allowed")
        if any(k in n for k in ("embed_tokens", "lm_head")):
            raise ValueError(f"{n}: embedding/output editing is not allowed")
        arrs[n + "::idx"] = idx
        arrs[n + "::val"] = newf[idx].numpy()
        man[n] = dict(shape=list(base_state[n].shape), changed=int(len(idx)), numel=int(base.numel()))
    np.savez(os.path.join(out_dir, "edit.npz"), **arrs)
    h = hashlib.sha256(open(os.path.join(out_dir, "edit.npz"), "rb").read()).hexdigest()
    tot = sum(v["changed"] for v in man.values())
    json.dump(dict(tensors=man, total_changed=tot, sha256_npz=h), open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return man

def apply_edit_to_policy(policy, edit_dir):
    man = json.load(open(os.path.join(edit_dir, "manifest.json")))
    z = np.load(os.path.join(edit_dir, "edit.npz"))
    assert hashlib.sha256(open(os.path.join(edit_dir, "edit.npz"), "rb").read()).hexdigest() == man["sha256_npz"], "hash mismatch"
    sd = dict(policy.named_parameters())
    sd.update(dict(policy.named_buffers()))
    with torch.no_grad():
        for n in man["tensors"]:
            p = sd[n]
            flat = p.data.view(-1)
            idx = torch.from_numpy(z[n + "::idx"]).to(flat.device)
            flat[idx] = torch.from_numpy(z[n + "::val"]).to(flat.device, flat.dtype)
    return man["total_changed"]

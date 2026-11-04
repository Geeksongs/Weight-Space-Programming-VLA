<h1 align="center">Weight-Space Programming for VLAs</h1>
<p align="center"><b>Fix a robot policy by editing its weights. No training. No gradients. One number.</b></p>

---

## One Number, 42% → 64%

A robot policy that fails the same way over and over doesn't need retraining. We took the official **SmolVLA**
(`lerobot/smolvla_libero`), found the one thing it kept doing wrong by watching its own rollouts, and fixed it by changing
**a single number in its weights**: no data, no GPU hours, no gradients.

| | success over 100 episodes |
|---|---|
| Original SmolVLA | 42 / 100 (**42%**) |
| **+ weight edit (B25)** | 64 / 100 (**64%**) &nbsp;**+22 pts** |
| **+ weight edit (B35)** | 64 / 100 (**64%**) &nbsp;**+22 pts** |

Task: `libero_spatial` #8, *"pick up the black bowl next to the plate and place it on the plate."* Both edits vs. original: Fisher exact p = 0.003.

- **0** training steps &nbsp;·&nbsp; **0** optimizer updates &nbsp;·&nbsp; **0** backward passes
- **1** number changed, in **1** tensor (`model.action_out_proj.bias`)
- **~600 bytes** for the whole model update (`edits/edit_B35/edit.npz`)
- Full edited weights on Hugging Face: [`pythonsong/smolvla-libero-wsp-B35`](https://huggingface.co/pythonsong/smolvla-libero-wsp-B35)

## The idea

Fine-tuning a VLA means data, GPUs and hours. But a policy's failure is often *mechanical*: it does one specific thing wrong, in
the same way, again and again. If you can read that failure off the policy's behaviour, you may be able to **program the fix
directly into the weights**, the way you would patch a few lines of code.

This repo is a small, concrete instance of that idea on a real VLA, following the *Weight-Space Programming* line of work from
MIT CSAIL ([blog](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs)).

## How it works: diagnose → write → verify

1. **Diagnose.** Roll the policy out on a handful of dev episodes and read only what the model itself can observe (its actions,
   gripper width, end-effector pose). Finding: failed episodes closed the gripper too far along x (≥ 4 cm in 3 of 5), and then
   re-grasped 5–15 times; successful ones closed at x ≤ 2 cm and grasped once.
2. **Write.** Shift the policy's x-action by a small constant, applied directly in the output layer's bias:

   ```python
   bias = weights["model.action_out_proj.bias"]
   bias[0] += 0.35        # B35  (0.25 for B25) — normalized action units
   ```

3. **Verify.** Evaluate original vs. edited weights on 100 episodes from init states the editor never saw.

The edit is stored as a tiny patch (`manifest.json` + `edit.npz`, with a SHA-256), so a "new model" is a few hundred bytes you
can diff, review, revert, or stack.

## Why this is interesting

- **Interpretable.** Every change is a named number in a named tensor, with a behavioural reason behind it.
- **Instant.** Patch → evaluate. No training loop, no data pipeline, no checkpoints to re-save.
- **Tiny.** The delta is bytes, not gigabytes.
- **Composable.** Patches are plain sparse diffs against the base weights.

## Reproduce

Environment: lerobot 0.4.3 (Python 3.10), LIBERO, one GPU. Paths in `code/` point at the original machine
(`/media/python/Seagate/...`); set `MODEL` in `code/eval_policy.py` and the `sys.path` / `LIBERO_CONFIG_PATH` lines first.

```bash
huggingface-cli download lerobot/smolvla_libero --local-dir models/smolvla_libero_official

# baseline
python code/eval_policy.py --suite libero_spatial --tasks 8 --episodes 40 --ep_offset 10 --out eval/base.json
# more seeds, OOM-safe parallel: python code/run_parallel.py --configs base,B25,B35 --start 10 --end 30 --shard 10 --seeds 2000,3000,4000
# edited
python code/eval_policy.py --suite libero_spatial --tasks 8 --episodes 40 --ep_offset 10 \
    --edit edits/edit_B35 --out eval/edited.json
```

`code/make_edit_bias.py NAME 0.35,0,0,0,0,0,0` regenerates the patch; `code/apply_edit.py` applies it and checks the hash.
Rollouts are stochastic (flow-matching sampling), so exact counts vary a little between runs.

## Repo map

| path | what |
|---|---|
| `edits/` | the weight patches (B15, B25, B35) |
| `code/` | evaluation, patch construction and application |
| `eval/` | raw per-episode logs (`sp8_test_*`, `ext/`) |
| `REPORT.md` | write-up |

## Scope

A proof of concept on one task (`libero_spatial` #8) with 100 episodes, using the official LIBERO-finetuned SmolVLA.

## Citation

```bibtex
@misc{song2026wspvla,
  title        = {Weight-Space Programming for VLAs},
  author       = {Song, Python},
  year         = {2026},
  howpublished = {\url{https://github.com/Geeksongs/Weight-Space-Programming-VLA}}
}
```

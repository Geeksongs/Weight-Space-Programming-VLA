"""OOM-safe parallel evaluation scheduler.

Splits (config x episode range) into shards and runs them as separate eval_policy.py processes.
 - Admission control: a new worker starts only if free GPU mem / free RAM leave headroom for one more
   worker (footprint measured from the already-running workers, conservative guess before that).
 - Watchdog: if system available RAM or free GPU mem drops below a hard floor, the newest worker is
   killed and its shard is re-queued (so the machine never swaps/hangs).
 - Failed shards (e.g. CUDA OOM) are retried up to 3 times.
Usage: python code/run_parallel.py --configs base,B25,B35 --start 50 --end 70 --shard 5 [--max_workers N]
"""
import argparse, os, subprocess, sys, time, json, collections
import psutil

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu_mem():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]).decode()
    used, total = [float(x) for x in out.strip().splitlines()[0].split(",")]
    return used, total  # MiB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="base,B25,B35")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=8)
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--end", type=int, default=70)
    ap.add_argument("--seeds", default="2000,3000,4000")
    ap.add_argument("--shard", type=int, default=5)
    ap.add_argument("--max_workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--gpu_reserve_mib", type=float, default=600)
    ap.add_argument("--ram_reserve_gb", type=float, default=3.0)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "eval", "ext"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    q = collections.deque()
    for off in range(a.start, a.end, a.shard):
        for sd in a.seeds.split(","):
            for c in a.configs.split(","):
                q.append(dict(cfg=c, off=off, seed=int(sd), n=min(a.shard, a.end - off), tries=0))
    running, done = [], []
    foot_gpu, foot_ram = 2500.0, 3.5 * 1024  # MiB, conservative guesses until measured
    t_last_start = 0

    def launch(j):
        tag = f"{j['cfg']}_s{j['seed']}_{j['off']}"
        cmd = [PY, os.path.join(ROOT, "code", "eval_policy.py"), "--suite", a.suite, "--tasks", str(a.task),
               "--episodes", str(j["n"]), "--ep_offset", str(j["off"]), "--seed_base", str(j["seed"]), "--out", os.path.join(a.outdir, tag + ".json")]
        if j["cfg"] != "base":
            cmd += ["--edit", os.path.join(ROOT, "edits", "edit_" + j["cfg"])]
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        log = open(os.path.join(a.outdir, tag + ".log"), "a")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        j["proc"], j["t0"], j["tries"] = p, time.time(), j["tries"] + 1
        running.append(j)
        print(f"[sched] start {tag} (try {j['tries']}) running={len(running)} queue={len(q)}", flush=True)

    while q or running:
        for j in running[:]:
            rc = j["proc"].poll()
            if rc is None:
                continue
            running.remove(j)
            tag = f"{j['cfg']}_s{j['seed']}_{j['off']}"
            ok = rc == 0 and os.path.exists(os.path.join(a.outdir, tag + ".json"))
            print(f"[sched] {tag} exit={rc} ok={ok} ({time.time()-j['t0']:.0f}s)", flush=True)
            if ok:
                done.append(j)
            elif j["tries"] < 3:
                q.appendleft(j)
            else:
                print(f"[sched] GIVING UP {tag}", flush=True)
        used, total = gpu_mem()
        avail_gb = psutil.virtual_memory().available / 2**30
        warm = [j for j in running if time.time() - j["t0"] > 60]
        if warm:
            foot_gpu = max(300.0, (used - 270) / len(running))  # 270 MiB = desktop baseline
            rss = sum(psutil.Process(j["proc"].pid).memory_info().rss for j in warm)
            foot_ram = max(500.0, rss / len(warm) / 2**20)
        # watchdog: hard floors -> kill newest, requeue
        if running and (avail_gb < a.ram_reserve_gb * 0.6 or total - used < a.gpu_reserve_mib * 0.4):
            j = running.pop()
            j["proc"].kill(); j["proc"].wait()
            j["tries"] -= 1  # a watchdog kill is not the shard's fault
            q.appendleft(j)
            print(f"[sched] WATCHDOG killed {j['cfg']}_s{j['seed']}_{j['off']} avail_ram={avail_gb:.1f}G gpu_free={total-used:.0f}MiB", flush=True)
            a.max_workers = max(1, len(running))  # don't exceed what just proved too much
            time.sleep(10)
            continue
        can_gpu = (total - used - a.gpu_reserve_mib) > foot_gpu * 1.15
        can_ram = (avail_gb - a.ram_reserve_gb) * 1024 > foot_ram * 1.15
        settled = time.time() - t_last_start > 25  # let the last worker's memory show up first
        if q and len(running) < a.max_workers and can_gpu and can_ram and settled:
            launch(q.popleft()); t_last_start = time.time()
            continue
        time.sleep(3)

    res = collections.defaultdict(dict)
    for j in done:
        r = json.load(open(os.path.join(a.outdir, f"{j['cfg']}_s{j['seed']}_{j['off']}.json")))["results"][str(a.task)]
        for k, s in enumerate(r["successes"]):
            res[j["cfg"]][(j["seed"], j["off"] + k)] = s
    json.dump({c: {f"seed{k[0]}_ep{k[1]}": v for k, v in sorted(d.items())} for c, d in res.items()}, open(os.path.join(a.outdir, "merged.json"), "w"), indent=1)
    for c, d in res.items():
        print(f"{c}: {sum(d.values())}/{len(d)}")


if __name__ == "__main__":
    main()

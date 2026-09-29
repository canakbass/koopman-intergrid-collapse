"""Neural ODE generalisation (HANDOFF 5-A). Two questions:
(A) does the silent inter-grid failure occur outside the linear Koopman model class, at the three
    rho values of Table 1? -> --model node --sampling regular, omega_mult in {0.75, 0.763932,
    2.763932}, seeds 0-4;
(B) does a second, incommensurate sampling rate on its own (12.5% of windows at dt2=sqrt(2)/2,
    no spectral lift) repair a Neural ODE? -> same grid with --sampling mr, seeds 0-4, plus
    omega_mult=0.5 so that rho=1/4 pairs with the existing regular runs in results_node5.jsonl.
Full diagnostics (no --light): kappa and loop_closure are needed. --dmd stays none -- the
multi-rate lift is Koopman-specific, it reinitialises model.omega and NODE has no omega.
Diagnostics are measured on the regular grid in both arms (run.py passes "regular" when
--sampling mr), so the two arms are directly comparable.

The 35 runs are split over two notebooks, because Kaggle allows two concurrent GPU sessions and
one notebook gets one GPU. The split is by alternating index, so each shard carries a mix of both
arms, all three rho values and all seeds: a shard that dies still leaves a balanced subset.
Each job is pinned to a GPU with CUDA_VISIBLE_DEVICES; the wrapper reads device_count() and scales
the pool to 3 jobs per GPU, so it is also correct if a machine turns out to have more than one."""
import base64, json, os

ACCOUNT = os.environ.get("KAGGLE_ACCOUNT", "anonymous")   # kullanici adi repoya yazilmaz (anonimlik)

SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py"]
WS = [0.75, 0.763932, 2.763932]
SHARDS = [("clock-node-gen", "kaggle_node_gen_a", "results_node_gen_a.jsonl"),
          ("clock-node-gen-b", "kaggle_node_gen_b", "results_node_gen_b.jsonl")]

jobs = []
for s in range(5):          # seed-major: an early stop still leaves both arms covered
    for w in WS:
        for smp in ["regular", "mr"]:
            jobs.append(["run.py", "--model", "node", "--sampling", smp,
                         "--omega_mult", str(w), "--seed", str(s)])
    jobs.append(["run.py", "--model", "node", "--sampling", "mr",
                 "--omega_mult", "0.5", "--seed", str(s)])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

for k, (slug, folder, out) in enumerate(SHARDS):
    shard = [j + ["--out", out] for j in jobs[k::len(SHARDS)]]
    kernel = f"""# {slug}: Neural ODE, three rho values, regular vs mixed-rate training, 5 seeds (shard {k + 1}/{len(SHARDS)}).
import base64, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
FILES = {files_b64!r}
JOBS = {shard!r}
OUT_FILES = {[out]!r}
W = "/kaggle/working/cd"; os.makedirs(W + "/ckpt", exist_ok=True); os.chdir(W)
for f, b in FILES.items(): open(f, "wb").write(base64.b64decode(b))
import torch
NG = max(1, torch.cuda.device_count())      # kac GPU verildiyse ona gore dagit
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", NG,
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
print(len(JOBS), "is,", NG, "GPU x 3 eszamanli", flush=True)
PY = sys.executable
t0 = time.time()
def go(ic):
    i, c = ic
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % NG))     # her is tek bir GPU gorur
    r = subprocess.run([PY] + c, capture_output=True, text=True, env=env)
    print(f"[{{time.time()-t0:6.0f}}s gpu{{i % NG}}]", " ".join(c),
          "OK" if r.returncode == 0 else "HATA\\n" + r.stderr[-1500:], flush=True)
    for f in OUT_FILES:     # her kosudan sonra kopyala: kernel yarida kesilirse de kismi sonuc kalir
        if os.path.exists(f): os.system(f"cp {{f}} /kaggle/working/")
with ThreadPoolExecutor(3 * NG) as ex: list(ex.map(go, enumerate(JOBS)))
print("BITTI", flush=True)
"""
    os.makedirs(folder, exist_ok=True)
    open(folder + "/kernel.py", "w").write(kernel)
    meta = {
        "id": f"{ACCOUNT}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    json.dump(meta, open(folder + "/kernel-metadata.json", "w"), indent=1)
    print(f"{len(shard):3d} jobs -> {folder}/ ({slug})")

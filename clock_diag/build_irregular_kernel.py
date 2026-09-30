"""Does the silent inter-grid failure need a regular grid? (reviewer objection: continuous-time
latent models are typically trained on irregularly sampled data.)

An increasing-irregularity ladder on the same task, all arms trained for 3000 iterations with the
full diagnostics: j5 (uniform jitter of +-0.05 on each step), j20 (+-0.20), exp (exponential
inter-arrival times clamped to [0.05, 3.0], i.e. no grid structure left). The regular arm is the
one already reported in Table 1 and in the Neural ODE table.

The diagnostics do not move with the training sampling: E^grid and E^mid are measured on a rollout
at Delta/16 against the Delta grid in every arm, so "is a model trained on irregular times accurate
between the grid points of Delta" stays a well-posed question. The oracle arm is the control on the
measurement itself -- its ratio must stay near 1 whatever the training sampling, and if it does not,
the ladder says nothing about the learned models.

Two kernels, because Kaggle allows two concurrent GPU sessions and the Neural ODE arm dominates the
cost (~19 min per run against ~2 min for a Koopman run).
"""
import base64, json, os

ACCOUNT = os.environ.get("KAGGLE_ACCOUNT", "anonymous")   # kullanici adi repoya yazilmaz (anonimlik)
SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py"]
MODES = ["j5", "j20", "exp"]

node_jobs, koop_jobs = [], []
for s in range(5):          # seed-major: an early stop still leaves every mode covered
    for mode in MODES:
        for w in [0.75, 0.763932]:
            node_jobs.append(["run.py", "--model", "node", "--sampling", mode,
                              "--omega_mult", str(w), "--seed", str(s), "--out", "results_irr_node.jsonl"])
        for w in [0.75, 0.763932, 2.763932]:
            for mdl in ["koop", "oracle"]:
                koop_jobs.append(["run.py", "--model", mdl, "--sampling", mode,
                                  "--omega_mult", str(w), "--seed", str(s), "--out", "results_irr_koop.jsonl"])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

SHARDS = [("clock-irr-node", "kaggle_irr_node", node_jobs, "results_irr_node.jsonl"),
          ("clock-irr-koop", "kaggle_irr_koop", koop_jobs, "results_irr_koop.jsonl")]

for slug, folder, jobs, out in SHARDS:
    kernel = f"""# {slug}: irregular-sampling ladder (j5 / j20 / exp), 5 seeds.
import base64, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
FILES = {files_b64!r}
JOBS = {jobs!r}
OUT_FILES = {[out]!r}
W = "/kaggle/working/cd"; os.makedirs(W + "/ckpt", exist_ok=True); os.chdir(W)
for f, b in FILES.items(): open(f, "wb").write(base64.b64decode(b))
import torch
NG = max(1, torch.cuda.device_count())
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", NG,
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
print(len(JOBS), "is,", NG, "GPU x 3 eszamanli", flush=True)
PY = sys.executable
t0 = time.time()
def go(ic):
    i, c = ic
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % NG))
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
    json.dump({"id": f"{ACCOUNT}/{slug}", "title": slug, "code_file": "kernel.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
               "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
               "competition_sources": [], "kernel_sources": []},
              open(folder + "/kernel-metadata.json", "w"), indent=1)
    print(f"{len(jobs):3d} jobs -> {folder}/ ({slug})")

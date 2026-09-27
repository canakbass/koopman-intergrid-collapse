"""Re-runs the seeds of Table 2 (0-1) and Table 3 treatment (0-2) that were produced with the
earlier right-eigenvector modal basis, using the current dmd_init.py (left-eigenvector basis),
so both tables are homogeneous in construction and device (T4 GPU). Arguments match the
original runs exactly (Table 3: --sampling mr --dmd mr --light; Table 2: regular, no --light)."""
import base64, json, os

SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py"]

jobs = []
for w in [0.75, 0.763932, 2.763932, 0.5]:
    for s in [0, 1, 2]:
        jobs.append(["run.py", "--model", "koop", "--sampling", "mr", "--dmd", "mr", "--light",
                     "--omega_mult", str(w), "--seed", str(s), "--out", "results_t3_rerun.jsonl"])
for w in [0.75, 0.763932, 2.763932]:
    for d in ["lock", "ft", "harm"]:
        for s in [0, 1]:
            jobs.append(["run.py", "--model", "koop", "--dmd", d, "--omega_mult", str(w), "--seed", str(s),
                         "--out", "results_t2_rerun.jsonl"])

OUT_FILES = ["results_t3_rerun.jsonl", "results_t2_rerun.jsonl"]
files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-rerun-leftbasis: Table 2 seeds 0-1 and Table 3 treatment seeds 0-2 with the current modal basis.
import base64, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
FILES = {files_b64!r}
JOBS = {jobs!r}
OUT_FILES = {OUT_FILES!r}
W = "/kaggle/working/cd"; os.makedirs(W + "/ckpt", exist_ok=True); os.chdir(W)
for f, b in FILES.items(): open(f, "wb").write(base64.b64decode(b))
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
r = subprocess.run([sys.executable, "dmd_init.py"], capture_output=True, text=True)
print("dmd_init selftest:", "OK" if r.returncode == 0 else "FAIL", r.stdout[-2000:], r.stderr[-1000:], flush=True)
PY = sys.executable
t0 = time.time()
def go(c):
    r = subprocess.run([PY] + c, capture_output=True, text=True)
    print(f"[{{time.time()-t0:6.0f}}s]", " ".join(c), "OK" if r.returncode == 0 else "HATA\\n" + r.stderr[-1500:], flush=True)
with ThreadPoolExecutor(3) as ex: list(ex.map(go, JOBS))
for f in OUT_FILES:
    if os.path.exists(f): os.system(f"cp {{f}} /kaggle/working/")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_rerun", exist_ok=True)
open("kaggle_rerun/kernel.py", "w").write(kernel)
meta = {
    "id": "anonymous/clock-rerun-leftbasis",
    "title": "clock-rerun-leftbasis",
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
json.dump(meta, open("kaggle_rerun/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "jobs packaged -> kaggle_rerun/")

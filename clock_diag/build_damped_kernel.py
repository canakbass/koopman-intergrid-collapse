"""Görev A Kaggle kernel paketleme: sönümlü/büyüyen özdeğer testi. 3 konfig (control/multirate/
oracle) x 3 tohum x 2 mu işareti (buyuyen +0.08 / sönümlü -0.08) = 18 iş."""
import base64, json, os

SRC_FILES = ["data.py", "models.py", "dmd_init.py", "models_robustness.py", "data_damped.py", "run_damped.py"]
CFGS = ["control", "multirate", "oracle"]
SEEDS = [0, 1, 2]
MUS = [0.08, -0.08]

jobs = []
for mu in MUS:
    for cfg in CFGS:
        for s in SEEDS:
            jobs.append(["run_damped.py", "--cfg", cfg, "--mu", str(mu), "--seed", str(s),
                         "--iters", "3000", "--dmd_at", "500", "--out", "damped_results.jsonl"])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-damped (tek cekirdek): Gorev A -- sonumlu/buyuyen (marjinal olmayan) ozdeger testi.
import base64, os, subprocess, sys, json, time
from concurrent.futures import ThreadPoolExecutor
FILES = {files_b64!r}
JOBS = {jobs!r}
W = "/kaggle/working/cd"; os.makedirs(W, exist_ok=True); os.chdir(W)
for f, b in FILES.items(): open(f, "wb").write(base64.b64decode(b))
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
PY = sys.executable
t0 = time.time()
def go(c):
    try:
        r = subprocess.run([PY] + c, capture_output=True, text=True, timeout=1700)
        ok = r.returncode == 0
        err = r.stderr[-1500:]
    except subprocess.TimeoutExpired:
        ok, err = False, "TIMEOUT after 1700s"
    print(f"[{{time.time()-t0:5.0f}}s]", " ".join(c), "OK" if ok else "HATA\\n" + err, flush=True)
with ThreadPoolExecutor(3) as ex: list(ex.map(go, JOBS))
os.system("cp damped_results.jsonl /kaggle/working/")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_damped", exist_ok=True)
open("kaggle_damped/kernel.py", "w").write(kernel)
meta = {
    "id": "anonymous/clock-damped",
    "title": "clock-damped",
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
json.dump(meta, open("kaggle_damped/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "iş paketlendi ->", "kaggle_damped/")

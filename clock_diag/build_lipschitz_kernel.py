"""Görev C Kaggle kernel paketleme: decoder spektral-norm (Lipschitz kısıtı) kontrolü.
dmd=none (control) konfigürasyonu, 3 tohum, omega_mult=0.763932 (results_merged2.jsonl'daki
SN'siz kontrolle AYNI omega -- doğrudan karşılaştırma için)."""
import base64, json, os

SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py", "models_robustness.py", "run_lipschitz.py"]
OMEGA = 0.763932
SEEDS = [0, 1, 2]

jobs = []
for s in SEEDS:
    jobs.append(["run_lipschitz.py", "--omega_mult", str(OMEGA), "--seed", str(s),
                 "--iters", "3000", "--out", "lipschitz_results.jsonl"])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-lipschitz (tek cekirdek): Gorev C -- decoder spektral-norm (Lipschitz) kontrolu.
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
os.system("cp lipschitz_results.jsonl /kaggle/working/")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_lipschitz", exist_ok=True)
open("kaggle_lipschitz/kernel.py", "w").write(kernel)
meta = {
    "id": "anonymous/clock-lipschitz",
    "title": "clock-lipschitz",
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
json.dump(meta, open("kaggle_lipschitz/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "iş paketlendi ->", "kaggle_lipschitz/")

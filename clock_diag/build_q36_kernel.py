"""Package a Kaggle kernel testing rho=1/3 (q=3) and rho=1/6, 5/6 (q=6), where the square
pixel lattice's 90-degree symmetry does not apply, to separate the two explanations for the
rho=1/4 (q=4) anomaly: pixel-lattice symmetry vs. general period-q reparametrization freedom."""
import base64, json, math, os

SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py"]
RHOS = [1/3, 1/6, 5/6]  # q=3, q=6, q=6 (numerator 5, mirrors the 5/4 companion to 1/4)
CONFIGS = [  # (model, variant, init, warmup, lam)
    ("oracle", "base", "small", 0, 0.01),
    ("koop", "base", "small", 0, 0.01),
    ("koop", "base", "band", 0, 0.01),
    ("koopns", "base", "band", 0, 0.01),
]
SEEDS = [0, 1]

jobs = []
for rho in RHOS:
    w = round(2 * rho, 6)  # omega_mult = omega/pi = 2*rho (Delta=1)
    for m, v, ini, wu, lam in CONFIGS:
        for s in SEEDS:
            jobs.append(["run.py", "--model", m, "--variant", v, "--init", ini, "--warmup", str(wu),
                         "--lam", str(lam), "--omega_mult", str(w), "--sampling", "regular", "--seed", str(s)])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-q36 (tek cekirdek): rho=1/3 (q=3) ve rho=1/6, 5/6 (q=6) testi.
# Amac: rho=1/4 (q=4) anomalisinin 90 derece piksel-izgara simetrisinden mi yoksa genel
# period-q reparametrizasyon serbestliginden mi kaynaklandigini ayirt etmek.
import base64, os, subprocess, sys, json, time
from concurrent.futures import ThreadPoolExecutor
FILES = {files_b64!r}
JOBS = {jobs!r}
W = "/kaggle/working/cd"; os.makedirs(W + "/ckpt", exist_ok=True); os.chdir(W)
for f, b in FILES.items(): open(f, "wb").write(base64.b64decode(b))
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
PY = sys.executable
t0 = time.time()
def go(c):
    r = subprocess.run([PY] + c, capture_output=True, text=True)
    print(f"[{{time.time()-t0:5.0f}}s]", " ".join(c), "OK" if r.returncode == 0 else "HATA\\n" + r.stderr[-1500:], flush=True)
with ThreadPoolExecutor(3) as ex: list(ex.map(go, JOBS))
os.system("cp results.jsonl /kaggle/working/")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_q36", exist_ok=True)
open("kaggle_q36/kernel.py", "w").write(kernel)
meta = {
    "id": "hcanakbass/clock-q36",
    "title": "clock-q36",
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
json.dump(meta, open("kaggle_q36/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "iş paketlendi ->", "kaggle_q36/", "(rho değerleri:", RHOS, ")")

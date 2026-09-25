"""Package a Kaggle kernel that runs the additional-seed sweep (seeds 2-5) for the three
rotation numbers behind Table 1 / the main 18-model claim. Mirrors the existing kaggle_*
kernel.py pattern: source files base64-embedded, JOBS run via ThreadPoolExecutor."""
import base64, json, math, os

SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py"]
GOLD = 2 * (3 - 5 ** 0.5) / 2  # rho_g / pi
RHOS = [0.75, round(GOLD, 6), round(GOLD + 2, 6)]
CONFIGS = [  # (model, variant, init, warmup, lam)
    ("oracle", "base", "small", 0, 0.01),
    ("koop", "base", "small", 0, 0.01),
    ("koop", "base", "band", 0, 0.01),
    ("koopns", "base", "band", 0, 0.01),
]
SEEDS = [2, 3, 4, 5]

jobs = []
for w in RHOS:
    for m, v, ini, wu, lam in CONFIGS:
        for s in SEEDS:
            jobs.append(["run.py", "--model", m, "--variant", v, "--init", ini, "--warmup", str(wu),
                         "--lam", str(lam), "--omega_mult", str(w), "--sampling", "regular", "--seed", str(s)])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-moreseeds (tek cekirdek): rho in {{3/8, rho_g, 1+rho_g}} icin ek tohumlar (seed 2-5),
# mevcut Table 1 konfigleri (oracle/small/band/no-static) icin seed sayisini 2'den 6'ya cikarir.
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
os.system("cp ckpt/*_portrait.npz /kaggle/working/ 2>/dev/null")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_moreseeds", exist_ok=True)
open("kaggle_moreseeds/kernel.py", "w").write(kernel)
meta = {
    "id": "anonymous/clock-moreseeds",
    "title": "clock-moreseeds",
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
json.dump(meta, open("kaggle_moreseeds/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "iş paketlendi ->", "kaggle_moreseeds/")

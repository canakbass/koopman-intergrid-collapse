"""Cylinder-flow benchmark (HANDOFF 5-B): does the silent inter-grid failure, and its multi-rate
repair, carry over from the rotating-sprite videos to a PDE?

Data: the vorticity field of a lattice-Boltzmann cylinder wake at Re=100 on its limit cycle, as a
Fourier series in shedding phase (cyl_lbm.py -> cyl_period.npz, embedded here; no solver runs on
Kaggle). The series is fitted to 0.18% residual and the mean field is dropped, so the model sees
the fluctuation field -- measured locally: on the raw field the oracle converges to exactly the
MSE of predicting the phase-averaged field (0.00582 both), i.e. it learns nothing but the temporal
mean, while on the fluctuation field the same configuration reaches 1e-4 with f_true=1.00.

Five arms x 5 seeds x 2 rho. rho = omega_mult/2 as in Table 1, and at both values the second and
third harmonics of the shedding frequency fold past Nyquist, which is the mechanism under test.
Full diagnostics (no --light): kappa is what identifies the clock regime. Validated locally at
omega_mult=0.75, seed 0: control reaches mid_over_grid=140.6 with kappa=3.94 while its grid loss
(4.9e-5) beats the oracle's (1.0e-4) -- the failure is invisible to any on-grid check.
"""
import base64, json, os

ACCOUNT = os.environ.get("KAGGLE_ACCOUNT", "anonymous")   # kullanici adi repoya yazilmaz (anonimlik)

SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py",
             "data_cyl.py", "run_cyl.py", "cyl_period.npz"]
OUT = "results_cyl.jsonl"
WS = [0.75, 0.763932]
ARMS = [["--model", "koop", "--sampling", "regular", "--dmd", "none"],    # olgu: sessiz ara-zaman hatasi
        ["--model", "koop", "--sampling", "mr", "--dmd", "none"],         # kontrol: ayni karisik veri, lift yok
        ["--model", "koop", "--sampling", "mr", "--dmd", "mr"],           # lift, gate'siz
        ["--model", "koop", "--sampling", "mr", "--dmd", "mrg"],          # lift, gate'li (tau=0.1)
        ["--model", "oracle", "--sampling", "regular", "--dmd", "none"]]  # referans olcek

jobs = []
for s in range(5):          # seed-major: an early stop still leaves every arm covered
    for w in WS:
        for arm in ARMS:
            jobs.append(["run_cyl.py"] + arm + ["--omega_mult", str(w), "--seed", str(s), "--out", OUT])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-cyl: cylinder wake (Re=100), five arms x 5 seeds x 2 rho.
import base64, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
FILES = {files_b64!r}
JOBS = {jobs!r}
OUT_FILES = {[OUT]!r}
W = "/kaggle/working/cd"; os.makedirs(W + "/ckpt", exist_ok=True); os.chdir(W)
for f, b in FILES.items(): open(f, "wb").write(base64.b64decode(b))
import torch
NG = max(1, torch.cuda.device_count())
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", NG,
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
import numpy as _np                  # verinin kimligi log'a gecsin (data_cyl import edilmiyor:
_m = _np.load("cyl_period.npz")       # ana surecte sys.path'te degil ve GPU bellegi ayirirdi)
print(f"cyl: T_sim={{float(_m['T']):.3f}} adim, harmonik={{int(_m['harmonics'])}}, periyodik model artigi={{float(_m['resid']):.4%}}", flush=True)
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

os.makedirs("kaggle_cyl", exist_ok=True)
open("kaggle_cyl/kernel.py", "w").write(kernel)
meta = {"id": f"{ACCOUNT}/clock-cyl", "title": "clock-cyl", "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
        "kernel_sources": []}
json.dump(meta, open("kaggle_cyl/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "jobs packaged -> kaggle_cyl/", f"({os.path.getsize('kaggle_cyl/kernel.py')//1024} KB)")

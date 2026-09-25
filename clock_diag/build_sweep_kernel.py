"""Görev D Kaggle kernel paketleme: α (karışım oranı) ve β=Δt2/Δt1 duyarlılık taraması, dmd=mr.
omega_mult=0.763932 (mevcut kaggle_out_gpu referansıyla AYNI omega).
α taraması: β=√2/2 sabit, α∈{0.03,0.0625,0.25} (0.125 referans zaten kaggle_out_gpu'da var).
β taraması: α=0.125 sabit, β∈{0.5,0.51 (Prop1'in kötü öngördüğü küçük-paydalı yakın değerler),
0.618 (altın oran eşleniği, 'iyi' irrasyonel-benzer), 0.85 ('iyi', √2/2'den farklı)}
(β=0.7071 referans zaten kaggle_out_gpu'da var). 3 tohum/değer = 9+12 = 21 iş."""
import base64, json, os

SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py", "data_sweep.py", "run_sweep.py"]
OMEGA = 0.763932
DT2_REF = 2 ** 0.5 / 2
ALPHAS = [0.03, 0.0625, 0.25]
BETAS = [0.5, 0.51, 0.618, 0.85]
SEEDS = [0, 1, 2]

jobs = []
for alpha in ALPHAS:
    for s in SEEDS:
        jobs.append(["run_sweep.py", "--omega_mult", str(OMEGA), "--mr_frac", str(alpha), "--dt2", str(DT2_REF),
                     "--seed", str(s), "--iters", "3000", "--dmd_at", "500", "--out", "sweep_results.jsonl"])
for beta in BETAS:
    for s in SEEDS:
        jobs.append(["run_sweep.py", "--omega_mult", str(OMEGA), "--mr_frac", "0.125", "--dt2", str(beta),
                     "--seed", str(s), "--iters", "3000", "--dmd_at", "500", "--out", "sweep_results.jsonl"])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-sweep (tek cekirdek): Gorev D -- alfa/beta duyarlilik taramasi, dmd=mr.
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
with ThreadPoolExecutor(4) as ex: list(ex.map(go, JOBS))
os.system("cp sweep_results.jsonl /kaggle/working/")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_sweep", exist_ok=True)
open("kaggle_sweep/kernel.py", "w").write(kernel)
meta = {
    "id": "anonymous/clock-sweep",
    "title": "clock-sweep",
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
json.dump(meta, open("kaggle_sweep/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "iş paketlendi ->", "kaggle_sweep/")

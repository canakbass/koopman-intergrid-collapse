"""Package a Kaggle kernel for FAZ 4: Duffing-tipi (sertlesen) acisal osilatorde r-bagimli
yerel sertifika testi, run_pend3.py'nin (FAZ 3, piksel sarkac/yumusayan) full/restricted
tasarimini birebir tekrarliyor. 3 tohum x 2 kol = 6 kosu."""
import base64, json, os

SRC_FILES = ["data.py", "models.py", "dmd_init_nonlinear.py", "data_pend.py", "data_duffing2.py", "run_pend4.py"]
ARMS = ["full", "restricted"]
SEEDS = [0, 1, 2]

jobs = []
for arm in ARMS:
    for s in SEEDS:
        jobs.append(["run_pend4.py", "--arm", arm, "--seed", str(s), "--iters", "2200",
                     "--sampling", "mr", "--out", "pend4_results.jsonl"])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-pend4 (tek cekirdek): FAZ 4 -- Duffing-tipi (sertlesen) acisal osilatorde
# r-bagimli yerel sertifika testi, FAZ 3'un (piksel sarkac) full/restricted tasarimiyla AYNI.
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
os.system("cp pend4_results.jsonl /kaggle/working/")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_pend4", exist_ok=True)
open("kaggle_pend4/kernel.py", "w").write(kernel)
meta = {
    "id": "anonymous/clock-pend4",
    "title": "clock-pend4",
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
json.dump(meta, open("kaggle_pend4/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "is paketlendi ->", "kaggle_pend4/")

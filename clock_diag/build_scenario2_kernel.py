"""Package a Kaggle kernel for the second synthetic scenario (two independent rotating objects):
control vs multi-rate-lifted vs oracle, 3 seeds, mirroring Table 3's structure."""
import base64, json, math, os

SRC_FILES = ["data.py", "data2.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run2.py"]
OMEGA1, OMEGA2 = 0.75, 0.382  # rotation numbers (x pi) of the two independent objects
SEEDS = [0, 1, 2]

jobs = []
for s in SEEDS:
    jobs.append(["run2.py", "--model", "oracle", "--omega1_mult", str(OMEGA1), "--omega2_mult", str(OMEGA2),
                 "--sampling", "regular", "--seed", str(s), "--iters", "3000"])
    jobs.append(["run2.py", "--model", "koop", "--dmd", "none", "--omega1_mult", str(OMEGA1), "--omega2_mult", str(OMEGA2),
                 "--sampling", "mr", "--seed", str(s), "--iters", "3000"])
    jobs.append(["run2.py", "--model", "koop", "--dmd", "mr", "--dmd_at", "500", "--omega1_mult", str(OMEGA1), "--omega2_mult", str(OMEGA2),
                 "--sampling", "mr", "--seed", str(s), "--iters", "3000"])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-scenario2 (tek cekirdek): iki bagimsiz donen nesne, control/multi-rate/oracle, 3 tohum.
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
    r = subprocess.run([PY] + c, capture_output=True, text=True)
    print(f"[{{time.time()-t0:5.0f}}s]", " ".join(c), "OK" if r.returncode == 0 else "HATA\\n" + r.stderr[-1500:], flush=True)
with ThreadPoolExecutor(3) as ex: list(ex.map(go, JOBS))
os.system("cp results2.jsonl /kaggle/working/")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_scenario2", exist_ok=True)
open("kaggle_scenario2/kernel.py", "w").write(kernel)
meta = {
    "id": "anonymous/clock-scenario2",
    "title": "clock-scenario2",
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
json.dump(meta, open("kaggle_scenario2/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "iş paketlendi ->", "kaggle_scenario2/")

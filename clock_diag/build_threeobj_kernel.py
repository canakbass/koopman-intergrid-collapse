"""Three independent rotating objects, control vs multi-rate lifting,
5 seeds. Tests whether the control/L_Delta ratio keeps shrinking monotonically (1->2->3 objects)
or saturates at 2. Mirrors the existing kaggle_*/build_*_kernel.py pattern exactly."""
import base64, json, os

SRC_FILES = ["data.py", "data2.py", "data_threeobj.py", "models.py", "losses.py", "dmd_init.py", "run3.py"]

jobs = []
for s in range(5):
    jobs.append(["run3.py", "--model", "koop", "--dmd", "none", "--sampling", "mr", "--seed", str(s),
                 "--out", "results3.jsonl"])
    jobs.append(["run3.py", "--model", "koop", "--dmd", "mr", "--dmd_at", "500", "--sampling", "mr",
                 "--seed", str(s), "--out", "results3.jsonl"])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-threeobj: uc bagimsiz donen nesne, control vs multi-rate, 5 tohum.
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
    print(f"[{{time.time()-t0:6.0f}}s]", " ".join(c), "OK" if r.returncode == 0 else "HATA\\n" + r.stderr[-1500:], flush=True)
with ThreadPoolExecutor(3) as ex: list(ex.map(go, JOBS))
if os.path.exists("results3.jsonl"): os.system("cp results3.jsonl /kaggle/working/")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_threeobj", exist_ok=True)
open("kaggle_threeobj/kernel.py", "w").write(kernel)
meta = {
    "id": "anonymous/clock-threeobj",
    "title": "clock-threeobj",
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
json.dump(meta, open("kaggle_threeobj/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "is paketlendi -> kaggle_threeobj/")

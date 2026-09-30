"""Is the repair's failure on the quasi-periodic system a capacity limit?

A frequency-modulated rotation has a pure point spectrum at the combination frequencies
m*omega + k*nu, so a linear Koopman generator can represent it given enough modes; expanding the
render's harmonics against the Bessel amplitudes J_k(mA) puts 90% of the oscillating energy in 13
modes, 95% in 22 and 99% in 57. The models of the main quasi-periodic table carry four oscillators.

This sweeps the oscillator budget over {4, 12, 24} for the phenomenon, the same-data control and the
gate-free lift, at two rotation numbers and five seeds. If the obstruction is the budget, the lift
should improve monotonically with it; if it does not, the explanation is wrong and the limit lies
elsewhere (the three-object check suggests separability after folding as the other candidate).
"""
import base64, json, os

ACCOUNT = os.environ.get("KAGGLE_ACCOUNT", "anonymous")   # kullanici adi repoya yazilmaz (anonimlik)
SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py",
             "data_qp.py", "run_qp.py", "run_qp_budget.py"]
OUT = "results_qp_budget.jsonl"
ARMS = [["--sampling", "regular", "--dmd", "none"],   # olgu
        ["--sampling", "mr", "--dmd", "none"],        # kontrol
        ["--sampling", "mr", "--dmd", "mr"]]          # lift, gate'siz

jobs = []
for s in range(5):
    for n in (4, 12, 24):
        for w in (0.75, 0.763932):
            for arm in ARMS:
                jobs.append(["run_qp_budget.py", "--model", "koop"] + arm +
                            ["--n_osc", str(n), "--omega_mult", str(w), "--seed", str(s), "--out", OUT])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}
kernel = f"""# clock-qp-budget: oscillator-budget sweep on the quasi-periodic system.
import base64, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
FILES = {files_b64!r}
JOBS = {jobs!r}
OUT_FILES = {[OUT]!r}
W = "/kaggle/working/cd"; os.makedirs(W + "/ckpt", exist_ok=True); os.chdir(W)
for f, b in FILES.items(): open(f, "wb").write(base64.b64decode(b))
import torch
NG = max(1, torch.cuda.device_count())
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", NG, flush=True)
print(len(JOBS), "is,", NG, "GPU x 3 eszamanli", flush=True)
PY = sys.executable
t0 = time.time()
def go(ic):
    i, c = ic
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % NG))
    r = subprocess.run([PY] + c, capture_output=True, text=True, env=env)
    print(f"[{{time.time()-t0:6.0f}}s gpu{{i % NG}}]", " ".join(c),
          "OK" if r.returncode == 0 else "HATA\\n" + r.stderr[-1500:], flush=True)
    for f in OUT_FILES:
        if os.path.exists(f): os.system(f"cp {{f}} /kaggle/working/")
with ThreadPoolExecutor(3 * NG) as ex: list(ex.map(go, enumerate(JOBS)))
shutil.rmtree(W + "/ckpt", ignore_errors=True)      # indirmeyi sisirmesin
print("BITTI", flush=True)
"""
os.makedirs("kaggle_qp_budget", exist_ok=True)
open("kaggle_qp_budget/kernel.py", "w").write(kernel)
json.dump({"id": f"{ACCOUNT}/clock-qp-budget", "title": "clock-qp-budget", "code_file": "kernel.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
           "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
          open("kaggle_qp_budget/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "jobs packaged -> kaggle_qp_budget/")

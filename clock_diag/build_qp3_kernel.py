"""Quasi-periodic rotation (data_qp.py): the combination-lattice test.

theta(t) = theta_0 + omega*t + A*sin(nu*t) with nu/omega irrational, so the trajectory never
repeats, the instantaneous angular velocity varies by +-31%, and the observable's spectrum is a set
of Bessel sidebands rather than a harmonic ladder. Two questions at once:
  (1) does the silent inter-grid failure need a periodic signal with harmonics?
  (2) does the phase-residual gate abstain on a second non-fixed-frequency system, independently of
      the pendulum? On the cylinder wake the gate had nothing to exclude, so it was not tested there.

There is no true oracle here: no single linear generator reproduces this flow. The reference arm is
an oracle initialised at the mean frequency omega, the same convention the pendulum table uses, and
it is a reference rather than a ceiling.
"""
import base64, json, os

ACCOUNT = os.environ.get("KAGGLE_ACCOUNT", "anonymous")   # kullanici adi repoya yazilmaz (anonimlik)
SRC_FILES = ["data.py", "models.py", "losses.py", "diagnostics.py", "dmd_init.py", "run.py",
             "data_qp.py", "run_qp.py"]
OUT = "results_qp3.jsonl"
ARMS = [["--model", "koop", "--sampling", "regular", "--dmd", "none"],    # olgu
        ["--model", "koop", "--sampling", "mr", "--dmd", "none"],         # kontrol
        ["--model", "koop", "--sampling", "mr", "--dmd", "mr"],           # lift, gate'siz
        ["--model", "koop", "--sampling", "mr", "--dmd", "mrg"],          # lift, gate'li: cekilmeli
        ["--model", "oracle", "--sampling", "regular", "--dmd", "none"]]  # ortalama frekansta referans

jobs = []
for s in range(5):
    for w in [0.75, 0.763932, 0.707107]:      # 0.707107 = sqrt2/4: irrasyonel ve dejenere DEGIL
        for arm in ARMS:
            jobs.append(["run_qp.py"] + arm + ["--omega_mult", str(w), "--seed", str(s), "--out", OUT])

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}
kernel = f"""# clock-qp3: quasi-periodic (frequency-modulated) rotation, five arms x 5 seeds x 2 rho.
import base64, os, subprocess, sys, time
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
import shutil                       # ckpt/ indirmeyi ~1 GB sisiriyor ve bu deneyde kullanilmiyor
shutil.rmtree(W + "/ckpt", ignore_errors=True)
print("BITTI", flush=True)
"""
os.makedirs("kaggle_qp3", exist_ok=True)
open("kaggle_qp3/kernel.py", "w").write(kernel)
json.dump({"id": f"{ACCOUNT}/clock-qp3", "title": "clock-qp3", "code_file": "kernel.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
           "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
          open("kaggle_qp3/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "jobs packaged -> kaggle_qp3/")

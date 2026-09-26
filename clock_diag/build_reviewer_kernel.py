"""Reviewer-response kernel packager (Gorev A/B/C/D). Mirrors the existing kaggle_*
build_*_kernel.py pattern: source files base64-embedded in one kernel.py, jobs run via
ThreadPoolExecutor(3) on a single T4. Does NOT modify any protected file (data.py,
models.py, dmd_init.py, losses.py, run.py, run_pend.py, run2.py, data2.py, diagnostics.py);
new behaviour lives entirely in new files (run_noise.py, run_pend_noise.py,
data_oneobj_wide.py, run_oneobj_wide.py) that only import the protected ones."""
import base64, json, os

SRC_FILES = ["data.py", "data2.py", "data_pend.py", "data_oneobj_wide.py", "models.py", "losses.py",
             "diagnostics.py", "dmd_init.py", "run.py", "run_pend.py", "run2.py",
             "run_oneobj_wide.py", "run_noise.py", "run_pend_noise.py"]

RHOS_SINGLE = [0.75, 0.763932, 2.763932]     # 3/8, rho_g, 1+rho_g
RHOS_LIFT = RHOS_SINGLE + [0.5]              # + 1/4 (tab:lifting's 4th row)

jobs = []

# ---- Gorev B / Table 2 (tab:single-rate): extra seeds on top of existing {0} (none) / {0,1} (lock/ft/harm)
for w in RHOS_SINGLE:
    for s in [1, 2, 3, 4]:
        jobs.append(["run.py", "--model", "koop", "--omega_mult", str(w), "--seed", str(s),
                     "--out", "results_t2_moreseeds.jsonl"])
    for d in ["lock", "ft", "harm"]:
        for s in [2, 3, 4]:
            jobs.append(["run.py", "--model", "koop", "--dmd", d, "--omega_mult", str(w), "--seed", str(s),
                         "--out", "results_t2_moreseeds.jsonl"])

# ---- Gorev B / Table 3 (tab:lifting): extra seeds {3,4} on top of existing {0,1,2}; --light matches original
for w in RHOS_LIFT:
    for s in [3, 4]:
        jobs.append(["run.py", "--model", "koop", "--sampling", "mr", "--light", "--omega_mult", str(w),
                     "--seed", str(s), "--out", "results_t3_moreseeds.jsonl"])
        jobs.append(["run.py", "--model", "koop", "--sampling", "mr", "--dmd", "mr", "--light",
                     "--omega_mult", str(w), "--seed", str(s), "--out", "results_t3_moreseeds.jsonl"])

# ---- Gorev B / Table 5 (tab:pendulum): extra seeds {3,4} (control/ungated/gated), {2,3,4} (oracle, was 2 seeds)
for s in [3, 4]:
    jobs.append(["run_pend.py", "--model", "koop", "--sampling", "mr", "--dmd", "none", "--seed", str(s),
                 "--out", "results_pend_moreseeds.jsonl"])
    jobs.append(["run_pend.py", "--model", "koop", "--sampling", "mr", "--dmd", "partial", "--dmd_at", "1500",
                 "--seed", str(s), "--out", "results_pend_moreseeds.jsonl"])
    jobs.append(["run_pend.py", "--model", "koop", "--sampling", "mr", "--dmd", "partialg", "--dmd_at", "1500",
                 "--seed", str(s), "--out", "results_pend_moreseeds.jsonl"])
for s in [2, 3, 4]:
    jobs.append(["run_pend.py", "--model", "oracle", "--sampling", "regular", "--dmd", "none", "--seed", str(s),
                 "--out", "results_pend_moreseeds.jsonl"])

# ---- Gorev A / eksen 1 (kapasite, tek nesne, n_osc=6 vs n_osc=4 -- n_osc=4 already covered by Table 3's
#      omega=0.75 rows above): koopns, same rho/sampling, 5 seeds, control + mr
for s in range(5):
    jobs.append(["run.py", "--model", "koopns", "--sampling", "mr", "--dmd", "none", "--omega_mult", "0.75",
                 "--light", "--seed", str(s), "--out", "results_axis1.jsonl"])
    jobs.append(["run.py", "--model", "koopns", "--sampling", "mr", "--dmd", "mr", "--dmd_at", "500",
                 "--omega_mult", "0.75", "--light", "--seed", str(s), "--out", "results_axis1.jsonl"])

# ---- Gorev A / eksen 2 (sahne, sabit kapasite n_osc=6,n_static=4 -- tek nesne, data2 render'i): 5 seeds
for s in range(5):
    jobs.append(["run_oneobj_wide.py", "--model", "koop", "--omega_mult", "0.75", "--sampling", "mr",
                 "--dmd", "none", "--seed", str(s), "--out", "results_oneobj_wide.jsonl"])
    jobs.append(["run_oneobj_wide.py", "--model", "koop", "--omega_mult", "0.75", "--sampling", "mr",
                 "--dmd", "mr", "--dmd_at", "500", "--seed", str(s), "--out", "results_oneobj_wide.jsonl"])

# ---- Gorev A / iki-nesne (Table 6) extra seeds {3,4} on existing {0,1,2}
for s in [3, 4]:
    jobs.append(["run2.py", "--model", "koop", "--dmd", "none", "--omega1_mult", "0.75", "--omega2_mult", "0.382",
                 "--sampling", "mr", "--seed", str(s), "--out", "results2_moreseeds.jsonl"])
    jobs.append(["run2.py", "--model", "koop", "--dmd", "mr", "--dmd_at", "500", "--omega1_mult", "0.75",
                 "--omega2_mult", "0.382", "--sampling", "mr", "--seed", str(s), "--out", "results2_moreseeds.jsonl"])

# ---- Gorev C: Neural ODE at rho=1/4, 5 seeds (seed 0 reproduces the paper's single-seed obs:regimes point)
for s in range(5):
    jobs.append(["run.py", "--model", "node", "--omega_mult", "0.5", "--seed", str(s),
                 "--out", "results_node5.jsonl"])

# ---- Gorev D: noise sweep, rigid rotation (rho_g), dmd=mr, gate-free residual/margin readout
for noise in [0.01, 0.03, 0.05]:
    for s in [0, 1, 2]:
        jobs.append(["run_noise.py", "--model", "koop", "--sampling", "mr", "--dmd", "mr", "--dmd_at", "500",
                     "--omega_mult", "0.763932", "--noise", str(noise), "--light", "--seed", str(s)])

# ---- Gorev D: noise sweep, pixel pendulum, gated partial lock (false-accept probe)
for noise in [0.01, 0.03, 0.05]:
    for s in [0, 1, 2]:
        jobs.append(["run_pend_noise.py", "--model", "koop", "--sampling", "mr", "--dmd", "partialg",
                     "--dmd_at", "1500", "--noise", str(noise), "--seed", str(s)])

OUT_FILES = ["results_t2_moreseeds.jsonl", "results_t3_moreseeds.jsonl", "results_pend_moreseeds.jsonl",
             "results_axis1.jsonl", "results_oneobj_wide.jsonl", "results2_moreseeds.jsonl",
             "results_node5.jsonl", "results_noise.jsonl", "results_pend_noise.jsonl"]

files_b64 = {f: base64.b64encode(open(f, "rb").read()).decode() for f in SRC_FILES}

kernel = f"""# clock-reviewer-response (tek cekirdek): Gorev A/B/C/D -- ek tohumlar (Table 2/3/5/6),
# tek/iki-nesne kapasite-vs-sahne ayrimi, Neural ODE 5 tohum, gurultu/esik taramasi.
import base64, os, subprocess, sys, json, time
from concurrent.futures import ThreadPoolExecutor
FILES = {files_b64!r}
JOBS = {jobs!r}
OUT_FILES = {OUT_FILES!r}
W = "/kaggle/working/cd"; os.makedirs(W + "/ckpt", exist_ok=True); os.chdir(W)
for f, b in FILES.items(): open(f, "wb").write(base64.b64decode(b))
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
r = subprocess.run([sys.executable, "dmd_init.py"], capture_output=True, text=True)
print("dmd_init selftest:", "OK" if r.returncode == 0 else "FAIL", r.stdout[-2000:], r.stderr[-1000:], flush=True)
PY = sys.executable
t0 = time.time()
def go(c):
    r = subprocess.run([PY] + c, capture_output=True, text=True)
    print(f"[{{time.time()-t0:6.0f}}s]", " ".join(c), "OK" if r.returncode == 0 else "HATA\\n" + r.stderr[-1500:], flush=True)
with ThreadPoolExecutor(3) as ex: list(ex.map(go, JOBS))
for f in OUT_FILES:
    if os.path.exists(f): os.system(f"cp {{f}} /kaggle/working/")
print("BITTI", flush=True)
"""

os.makedirs("kaggle_reviewer", exist_ok=True)
open("kaggle_reviewer/kernel.py", "w").write(kernel)
meta = {
    "id": "anonymous/clock-reviewer-response",
    "title": "clock-reviewer-response",
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
json.dump(meta, open("kaggle_reviewer/kernel-metadata.json", "w"), indent=1)
print(len(jobs), "is paketlendi -> kaggle_reviewer/")

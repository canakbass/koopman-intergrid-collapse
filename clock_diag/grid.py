import itertools, subprocess, json, os
from concurrent.futures import ThreadPoolExecutor
P = os.path.expanduser("~/Projeler/phyCSI_meta/.venv/bin/python")
jobs = []   # (model, variant, init, warmup, lam, ω, seed) — hepsi düzenli örnekleme
for w, s in [(w, s) for w in [0.5, 2.5] for s in [0, 1]]:
    jobs += [("oracle", "base", "small", 0, 0.01, w, s), ("koop", "base", "small", 0, 0.01, w, s),
             ("koop", "base", "band", 0, 0.01, w, s), ("koop", "both", "small", 1000, 0.001, w, s),
             ("koop", "both", "band", 1000, 0.001, w, s), ("koopns", "base", "band", 0, 0.01, w, s)]
jobs += [("node", "base", "small", 0, 0.01, w, 0) for w in [0.5, 2.5]]
done = set()
if os.path.exists("results.jsonl"):
    for l in open("results.jsonl"):
        d = json.loads(l); done.add((d["model"], d["variant"], d["init"], d["warmup"], d["lam"], d["omega_mult"], d["seed"]))
jobs = [j for j in jobs if j not in done]
print(len(jobs), "iş", flush=True)
def go(j):
    m, v, ini, wu, lam, w, s = j
    r = subprocess.run([P, "run.py", "--model", m, "--variant", v, "--init", ini, "--warmup", str(wu), "--lam", str(lam),
                        "--omega_mult", str(w), "--sampling", "regular", "--seed", str(s)],
                       capture_output=True, text=True)
    print(j, "OK" if r.returncode == 0 else "HATA " + r.stderr[-800:], flush=True)
with ThreadPoolExecutor(3) as ex: list(ex.map(go, jobs))
print("BİTTİ", flush=True)

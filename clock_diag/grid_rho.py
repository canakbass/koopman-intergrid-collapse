import itertools, subprocess, json, os
from concurrent.futures import ThreadPoolExecutor
P = os.path.expanduser("~/Projeler/phyCSI_meta/.venv/bin/python")
jobs = []   # dönme sayısı testi: ρ=ωΔ/2π rasyonel (3/8) vs irrasyonel (altın oran), düzenli örnekleme
GOLD = 2 * (3 - 5 ** 0.5) / 2          # ω/π, ρ = 0.381966
for w in [0.75, round(GOLD, 6), round(GOLD + 2, 6)]:
    for s in [0, 1]:
        jobs += [("oracle", "base", "small", 0, 0.01, w, s), ("koop", "base", "small", 0, 0.01, w, s),
                 ("koop", "base", "band", 0, 0.01, w, s), ("koopns", "base", "band", 0, 0.01, w, s)]
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

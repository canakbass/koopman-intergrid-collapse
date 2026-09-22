# Dönme sayısı testi: ρ = ωΔ/2π rasyonel (1/4, 3/8, 5/4) vs irrasyonel (altın, 1+altın)
import json, math, numpy as np
from collections import defaultdict
PI = math.pi
R = [json.loads(l) for l in open("results.jsonl")]
G = defaultdict(list)
for r in R: G[(r["model"], r["init"], r["variant"], round(r["omega_mult"], 4))].append(r)
m = lambda rs, k: float(np.mean([r[k] for r in rs]))
CF = [("oracle", "small", "oracle"), ("koop", "small", "koop küçük-init"), ("koop", "band", "koop band-init"), ("koopns", "band", "statiksiz band")]
WS = [(0.5, "1/4 (rasyonel)"), (0.75, "3/8 (rasyonel)"), (0.763932, "0.382 (altın, irr.)"), (2.5, "5/4 (rasyonel)"), (2.763932, "1.382 (altın, irr.)")]
print(f"{'konfig':16s} {'ω/π':>6s} {'ρ':>20s} {'n':>2s} | {'train':>7s} {'Δ/2 MSE':>17s} {'Δ/2÷train':>9s} {'f_true':>6s} {'f_alias':>7s} {'enc_err ı/o':>12s} {'ood_orta':>8s}")
for w, lab in WS:
    for mo, ini, name in CF:
        rs = G.get((mo, ini, "base", round(w, 4)))
        if not rs: continue
        per = "/".join(f"{r['mse_half']:.4f}" for r in rs)
        print(f"{name:16s} {w:6.3f} {lab:>20s} {len(rs):2d} | {m(rs,'train_mse'):7.4f} {per:>17s} {m(rs,'mse_half')/m(rs,'train_mse'):9.0f} "
              f"{m(rs,'f_true'):6.2f} {m(rs,'f_alias'):7.2f} {m(rs,'enc_err_grid'):5.2f}/{m(rs,'enc_err_mid'):<6.2f} {m(rs,'ood_mid'):8.2f}")
    print()

# DMD yeniden başlatma sonuçları: sprite (ρ taraması) + sarkaç
import json, math, os, numpy as np
from collections import defaultdict
PI = math.pi
R = [json.loads(l) for l in open("results.jsonl")] if os.path.exists("results.jsonl") else []
G = defaultdict(list)
for r in R: G[(r["model"], r.get("dmd", "none"), round(r["omega_mult"], 4))].append(r)
WS = [(0.5, "1/4"), (0.75, "3/8"), (0.763932, "0.382 irr"), (2.763932, "1.382 irr (Nyquist üstü)")]
CF = [("koop", "none", "taban (küçük-init)"), ("oracle", "none", "oracle (ω₀=gerçek)"), ("koop", "lock", "DMD kilitli"),
      ("koop", "ft", "DMD + ince ayar"), ("koop", "harm", "DMD + harmonik kaldırma")]
print("## Sprite, düzenli örnekleme — hedef: Δ/2 MSE ~1e-4 (oracle), f_true → 1")
print(f"{'konfig':24s} {'ρ':>24s} | {'train':>7s} {'Δ/2 MSE (tohum)':>17s} {'f_true':>6s} {'f_alias':>7s} | {'DMD bulunan |ω|/π':>26s} {'son |ω|/π':>26s} {'enc ı/o':>9s} {'ood':>5s}")
for w, lab in WS:
    for mo, dm, name in CF:
        rs = G.get((mo, dm, round(w, 4)))
        if not rs: continue
        mm = lambda k: float(np.mean([r[k] for r in rs]))
        per = "/".join(f"{r['mse_half']:.4f}" for r in rs)
        dmdw = ",".join(f"{abs(x)/PI:.2f}" for x in rs[0].get("dmd_omegas", [])) or "-"
        lw = ",".join(f"{abs(x)/PI:.2f}" for x in (rs[0].get("learned_omegas") or []))
        print(f"{name:24s} {lab:>24s} | {mm('train_mse'):7.4f} {per:>17s} {mm('f_true'):6.2f} {mm('f_alias'):7.2f} | {dmdw:>26s} {lw:>26s} "
              f"{mm('enc_err_grid'):4.2f}/{mm('enc_err_mid'):<4.2f} {mm('ood_mid'):5.2f}")
    print()
if os.path.exists("results_pend.jsonl"):
    print(f"## Piksel sarkaç (ω0={2.4}, ρ≈0.382, durgun bırakma, genlik 0.3–1.2 rad), tek koşu")
    print(f"{'konfig':24s} | {'train':>7s} {'Δ/2 MSE':>8s} {'3Δ/7 MSE':>8s} {'Δ/2÷train':>9s} {'f_true(8°)':>10s} | {'DMD |ω|':>22s} {'son |ω|':>22s}")
    for l in open("results_pend.jsonl"):
        r = json.loads(l); name = {"none": r["model"], "lock": "DMD kilitli", "ft": "DMD + ince ayar", "harm": "DMD + harmonik"}[r["dmd"]] if r["model"] != "oracle" else "oracle (ω₀=2.4)"
        if r["model"] == "koop" and r["dmd"] == "none": name = "taban (küçük-init)"
        print(f"{name:24s} | {r['mse_train']:7.4f} {r['mse_half']:8.4f} {r['mse_3_7']:8.4f} {r['mse_half']/r['mse_train']:9.1f} {r['f_true']:10.2f} | "
              f"{','.join(f'{abs(x):.2f}' for x in r.get('dmd_omegas', [])) or '-':>22s} {','.join(f'{abs(x):.2f}' for x in r['learned_omegas']):>22s}")

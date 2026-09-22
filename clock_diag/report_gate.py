# Eşikli spektral kilitleme: tek tablo. Kontrol = aynı karışık veri, DMD yok (kaggle_out_partial).
import json, glob, math
PI = math.pi
def load(root, n):
    seen, out = set(), []
    for f in glob.glob(f"{root}/**/{n}", recursive=True):
        for l in open(f):
            if l not in seen: seen.add(l); out.append(json.loads(l))
    return out
pix = [json.loads(l) for l in open(glob.glob("kaggle_out_gate/**/pixel_dmd.json", recursive=True)[0])][0] if glob.glob("kaggle_out_gate/**/pixel_dmd.json", recursive=True) else []
print("## Piksel DMD teşhisi (encoder yok, PCA-24; Δt2 artığı < 0.1 ⇒ doğrusal-tutarlı mod var)")
for p in pix:
    print(f"{p['case']:34s} bant içi en iyi ω={p['band_best_omega']:+.3f} rad/s  artık={p['band_best_residual']:.3f}  | mod artıkları={p['residual']}")
print("\n## Sarkaç (karışık veri; DMD 1500. adımda)")
print(f"{'konfig':28s} {'tohum':>5s} | {'train':>8s} {'Δ/2 MSE':>8s} {'f_true':>6s} | {'bant ω':>7s} {'artık':>6s} {'kilit?':>7s}")
ctrl = [r for r in load("kaggle_out_partial", "results_pend.jsonl") if r["dmd"] == "none"]
for r in sorted(ctrl, key=lambda r: r["seed"]):
    print(f"{'kontrol (DMD yok)':28s} {r['seed']:5d} | {r['mse_train']:8.5f} {r['mse_half']:8.4f} {r['f_true']:6.2f} | {'-':>7s} {'-':>6s} {'-':>7s}")
for d, name in [("partial", "kısmi kilit, eşiksiz"), ("partialg", "kısmi kilit, eşik 0.1")]:
    for r in sorted([r for r in load("kaggle_out_gate", "results_pend.jsonl") if r["dmd"] == d], key=lambda r: r["seed"]):
        lock = "iptal" if r.get("gated_skip") else "evet"
        print(f"{name:28s} {r['seed']:5d} | {r['mse_train']:8.5f} {r['mse_half']:8.4f} {r['f_true']:6.2f} | {r.get('partial_omega', float('nan')):+7.2f} {r.get('partial_residual', float('nan')):6.3f} {lock:>7s}")
print("\n## Sprite (karışık veri; eşikli mr-DMD, 500. adım) — referans: kontrol Δ/2 0.016–0.047, oracle 1e-4")
print(f"{'ρ':>6s} | {'Δ/2 MSE':>8s} {'3Δ/7 MSE':>8s} {'f_true':>6s} | {'mod artıkları':>36s} {'kilitlenen modlar':>18s} {'kaldırılan |ω|/π':>26s}")
for r in sorted(load("kaggle_out_gate", "results.jsonl"), key=lambda r: r["omega_mult"]):
    lab = {0.75: "3/8", 0.763932: "0.382", 2.763932: "1.382"}.get(round(r["omega_mult"], 6), r["omega_mult"])
    passed = "iptal" if r.get("gated_skip") else "".join("✓" if p else "·" for p in r.get("gate_pass", []))
    print(f"{lab:>6s} | {r['mse_half']:8.4f} {r['mse_3_7']:8.4f} {r['f_true']:6.2f} | {str([round(x,3) for x in r.get('mr_residual', [])]):>36s} {passed:>18s} "
          f"{','.join(f'{abs(x)/PI:.2f}' for x in r.get('dmd_omegas', [])):>26s}")

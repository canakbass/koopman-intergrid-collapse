# Çok hızlı DMD + genliğe bağlı frekans sonuçları. Kullanım: python report_mr.py <klasör> (alt klasörlerdeki results*.jsonl birleşir)
import json, math, sys, glob, numpy as np
from collections import defaultdict
PI = math.pi; root = sys.argv[1] if len(sys.argv) > 1 else "kaggle_out_mr"
def _load(name):   # çekirdek çıktısı aynı dosyayı hem çalışma klasöründe hem kopyada tutuyor: tekilleştir
    seen, out = set(), []
    for f in glob.glob(f"{root}/**/{name}", recursive=True):
        for l in open(f):
            if l not in seen: seen.add(l); out.append(json.loads(l))
    return out
S, Pd = _load("results.jsonl"), _load("results_pend.jsonl")
G = defaultdict(list)
for r in S: G[(round(r["omega_mult"], 4), r.get("dmd", "none"))].append(r)
print("## Sprite — eğitim: %87.5 Δt1=1, %12.5 Δt2=√2/2 (her iki kol aynı veri). Referans: tek hızlı oracle Δ/2 ≈ 1e-4")
print(f"{'ρ (Δt1)':>10s} {'konfig':22s} | {'Δ/2 MSE (s0/s1/s2)':>22s} {'3Δ/7 MSE (s0/s1/s2)':>22s} {'f_true':>15s} | {'kaldırılan |ω|/π (s0 | s1 | s2)':>40s} {'marj min':>8s}")
for w, lab in [(0.75, "3/8"), (0.763932, "0.382"), (2.763932, "1.382"), (0.5, "1/4")]:
    for dm, name in [("none", "karışık veri (kontrol)"), ("mr", "karışık + mr-DMD")]:
        rs = sorted(G.get((round(w, 4), dm), []), key=lambda r: r["seed"])
        if not rs: continue
        j = lambda k, f="{:.4f}": "/".join(f.format(r[k]) for r in rs)
        lift = " | ".join(",".join(f"{abs(x)/PI:.2f}" for x in sorted(r.get("dmd_omegas", []), key=abs) if abs(x) > 1e-6) for r in rs) if dm == "mr" else "-"
        mg = min([m for r in rs for m in r.get("mr_margin", []) if m == m] or [float("nan")])
        print(f"{lab:>10s} {name:22s} | {j('mse_half'):>22s} {j('mse_3_7'):>22s} {j('f_true', '{:.2f}'):>15s} | {lift:>40s} {mg:8.2f}")
    print()
if Pd:
    print("## Piksel sarkaç (ω0=2.4, genlik 0.3–1.2 rad, durgun bırakma)")
    print(f"{'konfig':34s} {'tohum':>5s} | {'train':>7s} {'Δ/2 MSE':>8s} {'3Δ/7 MSE':>8s} {'f_true(8°)':>10s} | {'son ω (rad/s)':>28s}")
    order = [("koop", "regular", "none"), ("oracle", "regular", "none"), ("oracleamp", "regular", "none"),
             ("koop", "mr", "none"), ("koop", "mr", "mr"), ("koopamp", "mr", "mr")]
    names = {order[0]: "taban (tek hız)", order[1]: "oracle doğrusal (tek hız)", order[2]: "oracle + ω(r) (tek hız)",
             order[3]: "karışık veri (kontrol)", order[4]: "karışık + mr-DMD doğrusal", order[5]: "karışık + mr-DMD + ω(r)"}
    for key in order:
        for r in sorted([r for r in Pd if (r["model"], r.get("sampling", "regular"), r["dmd"]) == key], key=lambda r: r["seed"]):
            print(f"{names[key]:34s} {r['seed']:5d} | {r['mse_train']:7.4f} {r['mse_half']:8.4f} {r['mse_3_7']:8.4f} {r['f_true']:10.2f} | "
                  f"{','.join(f'{x:.2f}' for x in r['learned_omegas']):>28s}")

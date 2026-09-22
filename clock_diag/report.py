# results.jsonl -> 3 tablo (tohum ortalaması). Kullanım: python report.py
import json, math, numpy as np
from collections import defaultdict
PI = math.pi
R = [json.loads(l) for l in open("results.jsonl")]
CFG = [("oracle", "base", "small", 0, "oracle (ω₀=gerçek)"), ("koop", "base", "small", 0, "koop küçük-init"),
       ("koop", "base", "band", 0, "koop band-init"), ("koop", "both", "small", 1000, "koop+kısıt küçük"),
       ("koop", "both", "band", 1000, "koop+kısıt band"), ("koopns", "base", "band", 0, "koop statiksiz band"),
       ("node", "base", "small", 0, "NODE küçük-init")]
G = defaultdict(list)
for r in R: G[(r["model"], r["variant"], r["init"], r["warmup"], r["omega_mult"])].append(r)
m = lambda rs, k: float(np.nanmean([r[k] for r in rs])) if rs and k in rs[0] else float("nan")
rng = lambda rs, k: (max(r[k] for r in rs) - min(r[k] for r in rs)) / 2 if len(rs) > 1 else 0.0

def each():
    for w in [0.5, 2.5]:
        for mo, v, ini, wu, name in CFG:
            rs = G.get((mo, v, ini, wu, w))
            if rs: yield w, name, rs
        yield w, None, None

print("## 1) Patoloji izolasyonu — Jacobian / sızıntı  (düzenli örnekleme, Δ=1)")
print(f"{'konfig':22s} {'ω':>5s} {'n':>2s} | {'‖J·ż‖/‖∂x̂/∂t‖':>14s} {'ω_img/π':>8s} {'ω_lat/π':>8s} {'κ':>6s} {'döngü kap.':>10s} {'‖enc(dec z)−z‖':>14s}")
for w, name, rs in each():
    if name is None: print(); continue
    print(f"{name:22s} {w:4.1f}π {len(rs):2d} | {m(rs,'chain_ratio'):14.3f} {m(rs,'w_img')/PI:+8.2f} {m(rs,'w_lat')/PI:8.2f} "
          f"{m(rs,'kappa'):6.2f} {m(rs,'loop_closure'):10.2f} {m(rs,'enc_self'):14.2f}")

print("## 2) Spektrum + Δ/2 hatasının ayrıştırılması")
print(f"{'konfig':22s} {'ω':>5s} | {'Koop ω*/π':>9s} {'DMD z/π':>7s} {'DMD enc(x)/π':>12s} | {'E_ızg':>6s} {'E_orta':>6s} {'orta÷ızg':>8s} "
      f"{'zaman.payı':>10s} {'E_AE_orta':>9s} {'enc_err ızg/orta':>16s} {'ood_orta':>8s} {'keskinlik ı/o':>13s}")
for w, name, rs in each():
    if name is None: print(); continue
    kd = m(rs, "koop_dominant") / PI if "koop_dominant" in rs[0] else float("nan")
    print(f"{name:22s} {w:4.1f}π | {kd:9.2f} {m(rs,'dmd_pred')/PI:7.2f} {m(rs,'dmd_enc_true')/PI:12.2f} | {m(rs,'E_grid'):6.4f} {m(rs,'E_mid'):6.4f} "
          f"{m(rs,'mid_over_grid'):8.0f} {m(rs,'timing_share'):10.2f} {m(rs,'E_ae_mid'):9.4f} {m(rs,'enc_err_grid'):7.2f}/{m(rs,'enc_err_mid'):<8.2f} "
          f"{m(rs,'ood_mid'):8.2f} {m(rs,'sharp_grid'):6.2f}/{m(rs,'sharp_mid'):.2f}")

print("## 3) Ablasyon — ara zaman MSE ve spektral doğruluk (hedef: 0.5π'de ω, 2.5π'de alias 0.5π — düzenli örneklemede ulaşılabilir en iyi)")
print(f"{'konfig':22s} {'ω':>5s} | {'train':>7s} {'Δ/2 MSE':>15s} {'3Δ/7 MSE':>8s} {'Δ/2÷oracle':>10s} | {'f_true':>6s} {'f_alias':>7s} {'|DMD−hedef|/π':>13s}")
for w, name, rs in each():
    if name is None: print(); continue
    orc = G.get(("oracle", "base", "small", 0, w), [])
    tgt = 0.5
    print(f"{name:22s} {w:4.1f}π | {m(rs,'train_mse'):7.4f} {m(rs,'mse_half'):8.4f}±{rng(rs,'mse_half'):.4f} {m(rs,'mse_3_7'):8.4f} "
          f"{m(rs,'mse_half')/m(orc,'mse_half') if orc else float('nan'):10.2f} | {m(rs,'f_true'):6.2f} {m(rs,'f_alias'):7.2f} "
          f"{abs(m(rs,'dmd_pred')/PI - tgt):13.2f}")

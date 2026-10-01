"""Asserts every number paper_v2 states about the two new experiments against the raw jsonl files.
Run from clock_diag/. Any drift between the text and the data fails loudly instead of silently.
"""
import json, statistics as st

m = st.median
node = [json.loads(l) for p in ["kaggle_out_node_gen/results_node_gen_a.jsonl",
                                "kaggle_out_node_gen_b/results_node_gen_b.jsonl",
                                "results_node5.jsonl"] for l in open(p)]
cyl = [json.loads(l) for l in open("kaggle_out_cyl/results_cyl.jsonl")]
ok, bad = [], []


def chk(name, got, want, tol=0.02):
    good = abs(got - want) <= tol * max(abs(want), 1e-12)
    (ok if good else bad).append(f"{name}: iddia {want}, veri {got:.4g}")


def sel(rs, **kw):
    v = [r for r in rs if all(r.get(k) == x for k, x in kw.items())]
    assert v, kw
    return v


# --- Neural ODE (5-A)
assert len(node) == 40, len(node)
for w, want in [(0.5, 93.5), (0.75, 272), (0.763932, 339), (2.763932, 74.1)]:
    chk(f"node regular rho={w/2:.3f} mid/grid", m([r["mid_over_grid"] for r in sel(node, omega_mult=w, sampling="regular")]), want)
for w, want in [(0.5, 8.77), (0.75, 36.7), (0.763932, 25.7), (2.763932, 137)]:
    chk(f"node mr rho={w/2:.3f} mid/grid", m([r["mid_over_grid"] for r in sel(node, omega_mult=w, sampling="mr")]), want)
chk("node regular f_true en buyuk medyan",
    max(m([r["f_true"] for r in sel(node, omega_mult=w, sampling="regular")]) for w in (0.5, 0.75, 0.763932, 2.763932)), 0.01, tol=0.25)
for w, want in [(0.5, 2.30), (0.75, 3.57), (0.763932, 3.61), (2.763932, 3.66)]:
    chk(f"node regular rho={w/2:.3f} kappa", m([r["kappa"] for r in sel(node, omega_mult=w, sampling="regular")]), want)
k = [r["kappa"] for r in sel(node, omega_mult=0.5, sampling="regular")]
c = [r["loop_closure"] for r in sel(node, omega_mult=0.5, sampling="regular")]
chk("rho=1/4 kappa min", min(k), 2.11); chk("rho=1/4 kappa max", max(k), 2.34)
chk("rho=1/4 C min", min(c), 1.77); chk("rho=1/4 C max", max(c), 1.91)

# --- silindir (5-B)
assert len(cyl) == 50, len(cyl)
for w, want in [(0.75, 113), (0.763932, 97)]:
    chk(f"cyl olgu rho={w/2:.3f} mid/grid", m([r["mid_over_grid"] for r in sel(cyl, omega_mult=w, model="koop", sampling="regular", dmd="none")]), want)
for w, want in [(0.75, 18.09), (0.763932, 25.45)]:
    chk(f"cyl kontrol rho={w/2:.3f} mid/grid", m([r["mid_over_grid"] for r in sel(cyl, omega_mult=w, model="koop", sampling="mr", dmd="none")]), want)
for dmd, wants in [("mr", (1.04, 1.04)), ("mrg", (1.02, 1.02))]:
    for w, want in zip((0.75, 0.763932), wants):
        chk(f"cyl lift {dmd} rho={w/2:.3f} medyan", m([r["mid_over_grid"] for r in sel(cyl, omega_mult=w, dmd=dmd)]), want)
for w, want in [(0.75, 1.00), (0.763932, 1.01)]:
    chk(f"cyl oracle rho={w/2:.3f}", m([r["mid_over_grid"] for r in sel(cyl, omega_mult=w, model="oracle")]), want)
for dmd, worst in (("mr", 1.80), ("mrg", 2.56)):
    v = [r["mid_over_grid"] for r in sel(cyl, dmd=dmd)]
    chk(f"lift {dmd} 1.1 altinda kosu", sum(x < 1.1 for x in v), 9, tol=0.0)
    chk(f"lift {dmd} en kotu", max(v), worst)
    chk(f"lift {dmd} kappa medyan", m([r["kappa"] for r in sel(cyl, dmd=dmd)]), 1.00)
    chk(f"lift {dmd} f_true=1.00 olan kosu", sum(r["f_true"] >= 0.9999 for r in sel(cyl, dmd=dmd)), 9, tol=0.0)
ctl = [r["mid_over_grid"] for r in sel(cyl, model="koop", sampling="mr", dmd="none")]
chk("kontrolun en iyi kosusu", min(ctl), 7.70)
chk("cyl olgu E_grid rho=3/8", m([r["E_grid"] for r in sel(cyl, omega_mult=0.75, model="koop", sampling="regular", dmd="none")]), 5.7e-5)
chk("cyl oracle E_grid rho=3/8", m([r["E_grid"] for r in sel(cyl, omega_mult=0.75, model="oracle")]), 5.9e-6)
chk("gate: tumden cekilen kosu", sum(bool(r.get("gated_skip")) for r in sel(cyl, dmd="mrg")), 0, tol=0.0)
chk("gate: mod reddeden kosu", sum(sum(r["gate_pass"]) < 4 for r in sel(cyl, dmd="mrg")), 3, tol=0.0)
chk("lift temel frekans", sel(cyl, omega_mult=0.75, dmd="mrg")[0]["dmd_omegas"][0], 2.3559)

# --- veri kaynagi (cyl_period.npz)
import numpy as np
d = np.load("cyl_period.npz")
chk("periyodik model artigi %", float(d["resid"]) * 100, 0.18, tol=0.03)
co = d["coef"]; M = int(d["harmonics"])
E = [float((co[2 * i - 1] ** 2 + co[2 * i] ** 2).sum()) for i in range(1, M + 1)]
for i, want in [(0, 84.5), (1, 8.2), (2, 6.4)]:
    chk(f"harmonik {i+1} enerji payi %", E[i] / sum(E) * 100, want, tol=0.02)

# --- appendix: cozucu ve veri kaynagi sayilari
D_CYL, U0, NY = 18.0, 0.1, 180
T = float(d["T"])
chk("appendix T (latis adimi)", T, 1010.02)
chk("appendix St", D_CYL / (T * U0), 0.178, tol=0.01)
chk("appendix nu", U0 * D_CYL / 100.0, 0.018)
chk("appendix tau", 3 * (U0 * D_CYL / 100.0) + 0.5, 0.554)
chk("appendix tikanma %", D_CYL / NY * 100, 10.0)
chk("appendix harmonik sayisi", M, 8, tol=0.0)
chk("appendix harmonik 4 payi %", E[3] / sum(E) * 100, 0.6, tol=0.2)

# --- iki gosterimin ayni kosulardan geldigi (app:rho capraz atfi)
mg2 = [json.loads(l) for l in open("results_merged2.jsonl")]
orc14 = sorted([r for r in mg2 if r["model"] == "oracle" and r["omega_mult"] == 0.5], key=lambda r: r["seed"])
chk("app:rho 131 = mse_half/train_mse (ortalamalarin orani)",
    sum(r["mse_half"] for r in orc14) / sum(r["train_mse"] for r in orc14), 131, tol=0.01)
_v = sorted(r["mid_over_grid"] for r in orc14)
chk("midpoint-only oran, en kucuk", _v[0], 239, tol=0.01)
chk("midpoint-only oran, en buyuk", _v[-1], 331, tol=0.01)
chk("midpoint-only oran, medyan", m([r["mid_over_grid"] for r in orc14]), 284, tol=0.01)

# --- ornekleme merdiveni (tab:irregular)
irr_k = [json.loads(l) for l in open("kaggle_out_irr_koop/results_irr_koop.jsonl")]
irr_n = [json.loads(l) for l in open("kaggle_out_irr_node/results_irr_node.jsonl")]
assert len(irr_k) == 90 and len(irr_n) == 30
mg2 = [json.loads(l) for l in open("results_merged2.jsonl")]
node_reg = [json.loads(l) for p2 in ("kaggle_out_node_gen/results_node_gen_a.jsonl",
                                     "kaggle_out_node_gen_b/results_node_gen_b.jsonl") for l in open(p2)]
def _r(rs): return m([x["mid_over_grid"] for x in rs])
for w, want in [(0.75, 109), (0.763932, 147), (2.763932, 172)]:     # regular, YALNIZ init=small
    v = [r for r in mg2 if r["model"] == "koop" and r["omega_mult"] == w and r["sampling"] == "regular"
         and r.get("init") == "small" and r.get("variant") == "base"]
    chk(f"merdiven koop regular rho={w/2:.3f}", _r(v), want)
for w, wants in [(0.75, (79.3, 12.0, 0.97)), (0.763932, (64.0, 13.2, 0.98)), (2.763932, (23.5, 16.5, 1.06))]:
    for mode, want in zip(("j5", "j20", "exp"), wants):
        chk(f"merdiven koop {mode} rho={w/2:.3f}", _r(sel(irr_k, model="koop", omega_mult=w, sampling=mode)), want)
for w, wants in [(0.75, (102, 4.90, 1.01)), (0.763932, (103, 7.63, 1.00))]:
    for mode, want in zip(("j5", "j20", "exp"), wants):
        chk(f"merdiven node {mode} rho={w/2:.3f}", _r(sel(irr_n, omega_mult=w, sampling=mode)), want)
for w, want in [(0.75, 272), (0.763932, 339), (2.763932, 74.1)]:
    chk(f"merdiven node regular rho={w/2:.3f}", _r(sel(node_reg, omega_mult=w, sampling="regular")), want)
_o = [r["mid_over_grid"] for r in irr_k if r["model"] == "oracle"]
chk("oracle duzensizde en kotu", max(_o), 1.06, tol=0.10)
chk("oracle duzensizde kappa", m([r["kappa"] for r in irr_k if r["model"] == "oracle"]), 1.00)
chk("j5 f_true (koop, rho=3/8)", m([r["f_true"] for r in sel(irr_k, model="koop", omega_mult=0.75, sampling="j5")]), 0.0, tol=0.0)
chk("exp kappa koop rho=3/8", m([r["kappa"] for r in sel(irr_k, model="koop", omega_mult=0.75, sampling="exp")]), 3.84)
chk("exp kappa node rho=3/8", m([r["kappa"] for r in sel(irr_n, omega_mult=0.75, sampling="exp")]), 1.99)

# --- yari-periyodik (tab:qp), 3 rotation number, duzeltilmis f_true penceresi
qp = [json.loads(l) for l in open("kaggle_out_qp3/results_qp3.jsonl")]
assert len(qp) == 75
def _q(w, key, f="mid_over_grid"):
    v = [r for r in qp if (r["model"], r["sampling"], r["dmd"]) == key and r["omega_mult"] == w]
    assert len(v) == 5, (w, key, len(v))
    return m([r[f] for r in v])
for w, want in [(0.75, 183), (0.763932, 252), (0.707107, 180)]:
    chk(f"qp olgu oran rho={w/2:.3f}", _q(w, ("koop", "regular", "none")), want, tol=0.03)
    chk(f"qp olgu f_true rho={w/2:.3f}", _q(w, ("koop", "regular", "none"), "f_true"), 0.0, tol=0.0)
# gate li kolun kontrole gore MUTLAK kazanci ve gate in kilitleme sikligi
for w, gain, locks in [(0.75, 1.87, 3), (0.763932, 3.51, 5), (0.707107, 1.00, 1)]:
    c = _q(w, ("koop", "mr", "none"), "E_mid"); g = _q(w, ("koop", "mr", "mrg"), "E_mid")
    chk(f"qp gate li kazanc rho={w/2:.3f}", c / g, gain, tol=0.04)
    v = [r for r in qp if r["dmd"] == "mrg" and r["omega_mult"] == w]
    chk(f"qp gate kilitleme sayisi rho={w/2:.3f}", sum(not r.get("gated_skip") for r in v), locks, tol=0.0)
# temiz irrasyonel rho da lift fayda saglamiyor: iddianin can alici noktasi
_c = _q(0.707107, ("koop", "mr", "none"), "E_mid"); _g = _q(0.707107, ("koop", "mr", "mrg"), "E_mid")
assert 0.9 < _c / _g < 1.2, "sqrt2/4 te gate li kol fayda saglamamali"

# --- kappa/oran iddiasi (ozet cumlelerin tablodan kopmamasi icin)
_lr = [(_q(w, ("koop", "mr", "mr")), _q(w, ("koop", "mr", "mr"), "kappa")) for w in (0.75, 0.763932, 0.707107)]
chk("qp lift oran en kucuk", min(r for r, _ in _lr), 77.6, tol=0.03)
chk("qp lift oran en buyuk", max(r for r, _ in _lr), 157.4, tol=0.03)
chk("qp lift kappa en kucuk", min(k for _, k in _lr), 0.48, tol=0.05)
chk("qp lift kappa=1.00 olan rho sayisi", sum(abs(k - 1.0) < 0.01 for _, k in _lr), 2, tol=0.0)

# --- mod cakisma sayimlari (dejenerasyon iddiasinin kaynagi)
import collections as _co
_G = (5 ** 0.5 - 1) / 2
def _alias_count(rho, comb):
    if comb:
        return len({round((rho * (mm + _G * kk)) % 1.0, 9) for mm in range(1, 9) for kk in range(-6, 7)})
    return len({round((rho * mm) % 1.0, 9) for mm in range(1, 25)})
chk("saf donme rho=3/8 alias", _alias_count(0.375, False), 8, tol=0.0)
chk("saf donme rho_g alias", _alias_count((3 - 5 ** 0.5) / 2, False), 24, tol=0.0)
chk("kombinasyon rho_g alias", _alias_count((3 - 5 ** 0.5) / 2, True), 32, tol=0.0)
chk("kombinasyon rho=3/8 alias", _alias_count(0.375, True), 104, tol=0.0)
chk("kombinasyon sqrt2/4 alias", _alias_count(2 ** 0.5 / 4, True), 104, tol=0.0)

# --- mod butcesi taramasi (tab:qp-budget) ve spektral sayim
bud = [json.loads(l) for l in open("kaggle_out_qp_budget/results_qp_budget.jsonl")]
assert len(bud) == 90
for r in bud: r["n"] = len(r["learned_omegas"])
def _b(w, key, n, f="mid_over_grid"):
    v = [r for r in bud if (r["sampling"], r["dmd"]) == key and r["omega_mult"] == w and r["n"] == n]
    assert len(v) == 5, (w, key, n, len(v))
    return m([r[f] for r in v])
for w, wants in [(0.75, (145, 89, 31)), (0.763932, (111, 108, 14))]:
    for n, want in zip((4, 12, 24), wants):
        chk(f"butce lift oran rho={w/2:.3f} n={n}", _b(w, ("mr", "mr"), n), want, tol=0.03)
# oran duserken MUTLAK hatalar artiyor: iddianin can alici noktasi
for w, lo, hi in [(0.75, 14.5, 1.95), (0.763932, 22.6, 2.63)]:
    chk(f"butce lift E_0 artisi rho={w/2:.3f}", _b(w, ("mr", "mr"), 24, "E_grid") / _b(w, ("mr", "mr"), 4, "E_grid"), lo, tol=0.03)
    chk(f"butce lift E_1/2 artisi rho={w/2:.3f}", _b(w, ("mr", "mr"), 24, "E_mid") / _b(w, ("mr", "mr"), 4, "E_mid"), hi, tol=0.03)
chk("butce lift f_true n=4", m([_b(w, ("mr", "mr"), 4, "f_true") for w in (0.75, 0.763932)]), 0.405, tol=0.05)
chk("butce lift f_true n=24", m([_b(w, ("mr", "mr"), 24, "f_true") for w in (0.75, 0.763932)]), 0.197, tol=0.30)
# kontrolun mutlak hatasi butceyle sabit -> bozulma lift'e ozgu
for w in (0.75, 0.763932):
    chk(f"butce kontrol E_1/2 sabit rho={w/2:.3f}",
        _b(w, ("mr", "none"), 24, "E_mid") / _b(w, ("mr", "none"), 4, "E_mid"), 1.0, tol=0.10)

# --- spektral mod sayimi (Jacobi-Anger + Bessel genlikleri)
import math as _math, numpy as _np, torch as _torch
import data as _data, data_qp as _dqp
def _Jk(k, x, N=4096):
    tau = _np.linspace(0, 2 * _np.pi, N, endpoint=False)
    return float(_np.mean(_np.cos(k * tau - x * _np.sin(tau))))
_NT = 1024
_ang = _torch.arange(_NT, dtype=_torch.float32) * (2 * _math.pi / _NT)
_X = _data.render(_torch.zeros(_NT, dtype=_torch.long, device=_data.DEV), _ang.to(_data.DEV))[:, 0].reshape(_NT, -1).cpu().numpy()
_Em = (_np.abs(_np.fft.rfft(_X, axis=0) / _NT) ** 2).sum(1)
_rows = sorted((_Em[mm] * _Jk(kk, mm * _dqp.A) ** 2 for mm in range(1, 25) for kk in range(-12, 13)), reverse=True)
_tot = sum(_rows)
for thr, want in ((0.90, 13), (0.95, 22), (0.99, 57)):
    acc = 0; cnt = 0
    for e in _rows:
        acc += e / _tot; cnt += 1
        if acc >= thr: break
    chk(f"spektrum: %{thr*100:.0f} enerji icin mod", cnt, want, tol=0.0)

print(f"DOGRULANAN: {len(ok)}")
for b in bad: print("  UYUSMUYOR ->", b)
print("SONUC:", "hepsi tutuyor" if not bad else f"{len(bad)} UYUSMAZLIK")

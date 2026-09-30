"""Emits the LaTeX tables for the sampling-regularity ladder and the quasi-periodic system, straight
from the raw jsonl files. Run from clock_diag/; writes into the directory given as the first
argument, or the working directory.

Comparability note that the tables depend on: every irregular run uses the default small
initialisation, so the regular row is restricted to init=small in results_merged2.jsonl. Taking all
koop records there instead mixes the band-init and no-static variants and inflates the regular row
by a factor of three to five.
"""
import json, os, statistics as st, sys

m = st.median
RHO = {0.75: r"$3/8$", 0.763932: r"$\rho_g$", 2.763932: r"$1+\rho_g$"}
MODES = [("regular", "regular"), ("j5", r"jitter $\pm0.05$"), ("j20", r"jitter $\pm0.20$"), ("exp", "exponential")]


def load(p):
    with open(p) as f: return [json.loads(l) for l in f]


def num(x):
    return f"{x:.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def cell(v):
    return (f"${num(m([r['mid_over_grid'] for r in v]))}$", f"${m([r['f_true'] for r in v]):.2f}$",
            f"${m([r['kappa'] for r in v]):.2f}$") if v else ("--", "--", "--")


def irr_table(out):
    reg = load("results_merged2.jsonl")
    irr_k = load("kaggle_out_irr_koop2/results_irr_koop.jsonl")
    irr_n = load("kaggle_out_irr_node/results_irr_node.jsonl")
    node_reg = [r for p in ("kaggle_out_node_gen/results_node_gen_a.jsonl",
                            "kaggle_out_node_gen_b/results_node_gen_b.jsonl") for r in load(p)]
    assert len(irr_k) == 90 and len(irr_n) == 30
    out.append(r"""\begin{table}[t]
\centering
\caption{\textbf{The failure needs a grid, and realistic jitter is not enough to dissolve one}
(medians over five seeds; $\Eoff{1/2}/\Eoff{0}$ as in
\cref{sec:experiments}). Training times are perturbed from the grid $\{k\Delta\}$ by uniform jitter
of $\pm0.05$ or $\pm0.20$, or replaced by exponential inter-arrival times, which removes the grid
entirely. The diagnostics are unchanged: every arm is still evaluated at the midpoints of the same
$\Delta$ grid. The regular rows use the small initialisation, matching the irregular runs.}
\label{tab:irregular}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llccccc}
\toprule
& & \multicolumn{2}{c}{Koopman} & \multicolumn{2}{c}{Neural ODE} & oracle \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}
$\rho$ & training times & ratio & $f_{\text{true}}$ & ratio & $f_{\text{true}}$ & ratio \\
\midrule""")
    for w in (0.75, 0.763932, 2.763932):
        for i, (mode, lab) in enumerate(MODES):
            ks = [r for r in (reg if mode == "regular" else irr_k)
                  if r["model"] == "koop" and r["omega_mult"] == w and r["sampling"] == mode
                  and (mode != "regular" or (r.get("init") == "small" and r.get("variant") == "base"))]
            ns = [r for r in (node_reg if mode == "regular" else irr_n)
                  if r["omega_mult"] == w and r["sampling"] == mode]
            os_ = [r for r in (reg if mode == "regular" else irr_k)
                   if r["model"] == "oracle" and r["omega_mult"] == w and r["sampling"] == mode]
            kr, kf, _ = cell(ks); nr, nf, _ = cell(ns); orr, _, _ = cell(os_)
            pre = r"\multirow{4}{*}{" + RHO[w] + "}" if i == 0 else ""
            out.append(f"{pre} & {lab} & {kr} & {kf} & {nr} & {nf} & {orr} \\\\")
        out.append(r"\midrule" if w != 2.763932 else r"\bottomrule")
    out.append("\\end{tabular}\n\\end{table}\n")


def qp_table(out):
    rs = load("kaggle_out_qp/results_qp.jsonl")
    assert len(rs) == 50
    arms = [(("koop", "regular", "none"), "regular, no lift"),
            (("koop", "mr", "none"), "mixed-rate control"),
            (("koop", "mr", "mr"), "multi-rate lift"),
            (("koop", "mr", "mrg"), "multi-rate lift + gate"),
            (("oracle", "regular", "none"), "mean-frequency reference")]
    out.append(r"""\begin{table}[t]
\centering
\caption{\textbf{A quasi-periodic system: the failure survives, the repair does not}
(medians over five seeds). The glyph's phase is $\theta_0+\omega t+A\sin(\nu t)$ with $\nu/\omega$
irrational, so the trajectory never repeats, the instantaneous angular velocity varies by $\pm31\%$,
and the spectrum is a set of Bessel sidebands rather than a harmonic ladder. There is no oracle
here: no single linear generator reproduces this motion, and the last row is a reference
initialised at the mean frequency, not a ceiling --- it is the worst arm in the table.}
\label{tab:qp}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llcccc}
\toprule
$\rho$ & arm & $\Eoff{1/2}/\Eoff{0}$ & $\kappa$ & $f_{\text{true}}$ & gate \\
\midrule""")
    for w in (0.75, 0.763932):
        for i, (key, lab) in enumerate(arms):
            v = [r for r in rs if (r["model"], r["sampling"], r["dmd"]) == key and r["omega_mult"] == w]
            r_, f_, k_ = cell(v)
            if key[2] == "mrg":
                nlock = [sum(x.get("gate_pass", [])) for x in v]
                skip = sum(bool(x.get("gated_skip")) for x in v)
                g = f"locks ${min(nlock)}$--${max(nlock)}$ of $4$; abstains ${skip}/5$"
            else:
                g = "--"
            pre = r"\multirow{5}{*}{" + RHO[w] + "}" if i == 0 else ""
            out.append(f"{pre} & {lab} & {r_} & {k_} & {f_} & {g} \\\\")
        out.append(r"\midrule" if w == 0.75 else r"\bottomrule")
    out.append("\\end{tabular}\n\\end{table}")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "."
    hdr = "% Bu dosya make_tables_v3.py tarafindan ham jsonl'den uretilir; elle duzenleme."
    for fn, build in (("table_irregular", irr_table), ("table_qp", qp_table)):
        o = [hdr]; build(o)
        with open(os.path.join(dest, fn + ".tex"), "w") as f: f.write("\n".join(o) + "\n")
        print("yazildi:", fn + ".tex")

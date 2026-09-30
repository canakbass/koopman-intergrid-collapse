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
    irr_k = load("kaggle_out_irr_koop/results_irr_koop.jsonl")
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
\caption{\textbf{A quasi-periodic system: the failure survives and the gate carries the repair}
(medians over five seeds). \emph{Read the absolute columns, not the ratio}: the gate-free lift is
level with the control on $\Eoff{1/2}$ while its ratio is three times larger, because lifting
improves $\Eoff{0}$ by a factor of $2.6$ and shrinks the denominator. The gated arm is the best
learned arm here on every absolute measure. The glyph's phase is $\theta_0+\omega t+A\sin(\nu t)$ with $\nu/\omega$
irrational, so the trajectory never repeats and the instantaneous angular velocity varies by
$\pm31\%$. The spectrum is still a point spectrum, at the combination frequencies $m\omega+k\nu$
rather than on a harmonic ladder, so a linear generator exists in principle; what the models here
do not have is enough modes for it ($10$--$17$ carry $90\%$ of the oscillating
energy depending on the glyph, against four oscillators). The last row is not an oracle but a reference initialised at the mean frequency, and
it is the worst arm in the table, which is why no ``oracle level'' anchors this table.}
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


def budget_table(out):
    rs = load("kaggle_out_qp_budget/results_qp_budget.jsonl")
    assert len(rs) == 90, len(rs)
    for r in rs: r["n"] = len(r["learned_omegas"])
    arms = [(("regular", "none"), "regular, no lift"), (("mr", "none"), "mixed-rate control"),
            (("mr", "mr"), "multi-rate lift")]
    out.append(r"""\begin{table}[t]
\centering
\caption{\textbf{Enlarging the oscillator budget does not repair the lift on the quasi-periodic
system} (medians over five seeds). The observable needs $10$--$17$ modes for $90\%$ of its
oscillating energy depending on the glyph, and the models of \cref{tab:qp} carry four, so capacity is a candidate explanation for the
failure of the repair there. It does not survive the test: the ratio falls with the budget, but only
because $\Eoff{0}$ degrades faster than $\Eoff{1/2}$ --- both absolute errors get worse and
$f_{\text{true}}$ falls with them. The control's absolute error is flat across the budget, so this
is specific to the lift rather than a general effect of model size at a fixed iteration count.}
\label{tab:qp-budget}
\footnotesize
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llcccc}
\toprule
$\rho$ & arm & $n_{\text{osc}}$ & $\Eoff{0}$ & $\Eoff{1/2}$ & ratio \\
\midrule""")
    for w in (0.75, 0.763932):
        for i, (key, lab) in enumerate(arms):
            for j, n in enumerate((4, 12, 24)):
                v = [r for r in rs if (r["sampling"], r["dmd"]) == key and r["omega_mult"] == w and r["n"] == n]
                assert len(v) == 5
                nm = lab if j == 0 else ""
                pre = r"\multirow{9}{*}{" + RHO[w] + "}" if (i == 0 and j == 0) else ""
                out.append(f"{pre} & {nm} & ${n}$ & ${sci(m([r['E_grid'] for r in v]))}$ & "
                           f"${sci(m([r['E_mid'] for r in v]))}$ & ${num(m([r['mid_over_grid'] for r in v]))}$ \\\\")
            if i < 2: out.append(r"\cmidrule(lr){2-6}")
        out.append(r"\midrule" if w == 0.75 else r"\bottomrule")
    out.append("\\end{tabular}\n\\end{table}")


def sci(x):
    return f"{x:.0e}".replace("e-0", "e-").replace("e-", r"\text{e-}")


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "."
    hdr = "% Bu dosya make_tables_v3.py tarafindan ham jsonl'den uretilir; elle duzenleme."
    for fn, build in (("table_irregular", irr_table), ("table_qp", qp_table), ("table_qp_budget", budget_table)):
        o = [hdr]; build(o)
        with open(os.path.join(dest, fn + ".tex"), "w") as f: f.write("\n".join(o) + "\n")
        print("yazildi:", fn + ".tex")

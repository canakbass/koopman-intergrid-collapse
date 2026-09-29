"""Emits the two new LaTeX tables of paper_v2 straight from the raw jsonl files, so no number in
the paper is transcribed by hand: Neural ODE generalisation (5-A) and the cylinder wake (5-B).
Run from clock_diag/. Writes table_node.tex and table_cyl.tex into the directory given as the
first argument, or into the working directory if none is given.
"""
import json, os, statistics as st, sys

m = st.median
NODE_SRC = ["kaggle_out_node_gen/results_node_gen_a.jsonl",
            "kaggle_out_node_gen_b/results_node_gen_b.jsonl", "results_node5.jsonl"]
CYL_SRC = "kaggle_out_cyl/results_cyl.jsonl"
RHO = {0.5: r"$1/4$", 0.75: r"$3/8$", 0.763932: r"$\rho_g$", 2.763932: r"$1+\rho_g$"}


def load(p):
    with open(p) as f: return [json.loads(l) for l in f]


def num(x):
    return f"{x:.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def sci(x):
    return f"{x:.0e}".replace("e-0", "e-").replace("e-", r"\text{e-}")


def stats(v):
    mg = [r["mid_over_grid"] for r in v]
    return (f"${sci(m([r['E_grid'] for r in v]))}$ & ${sci(m([r['E_mid'] for r in v]))}$ & "
            f"${num(m(mg))}$ [${num(min(mg))}$, ${num(max(mg))}$] & "
            f"${m([r['kappa'] for r in v]):.2f}$ & ${m([r['f_true'] for r in v]):.2f}$")


def node_table(out):
    rs = [r for p in NODE_SRC for r in load(p)]
    assert len(rs) == 40, len(rs)
    out.append(r"""\begin{table}[t]
\centering
\caption{\textbf{The silent inter-grid failure in a Neural ODE} (five seeds per cell; medians, with
the range over seeds in brackets). The latent generator is a learned vector field integrated with
RK4, not a linear Koopman operator, and the failure appears at every rotation number. Training a
fraction of windows at the second rate (\emph{mixed-rate}) reduces it but does not remove it; the
Koopman oracle reaches $\mathcal{E}_{1/2}/\mathcal{E}_{\Delta}\approx1$ at the same $\rho$.}
\label{tab:node}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llccccc}
\toprule
$\rho$ & sampling & $\mathcal{E}_{\Delta}$ & $\mathcal{E}_{1/2}$ &
$\mathcal{E}_{1/2}/\mathcal{E}_{\Delta}$ & $\kappa$ & $f_{\text{true}}$ \\
\midrule""")
    for w in (0.5, 0.75, 0.763932, 2.763932):
        for smp, lab in (("regular", "regular"), ("mr", "mixed-rate")):
            v = [r for r in rs if r["omega_mult"] == w and r["sampling"] == smp]
            assert len(v) == 5
            pre = r"\multirow{2}{*}{" + RHO[w] + "}" if smp == "regular" else ""
            out.append(f"{pre} & {lab} & {stats(v)} \\\\")
        out.append(r"\midrule" if w != 2.763932 else r"\bottomrule")
    out.append("\\end{tabular}%\n}\n\\end{table}\n")


def cyl_table(out):
    rs = load(CYL_SRC)
    assert len(rs) == 50, len(rs)
    arms = [(("koop", "regular", "none"), "regular, no lift"),
            (("koop", "mr", "none"), "mixed-rate control"),
            (("koop", "mr", "mr"), "multi-rate lift"),
            (("koop", "mr", "mrg"), "multi-rate lift + gate"),
            (("oracle", "regular", "none"), "oracle")]
    out.append(r"""\begin{table}[t]
\centering
\caption{\textbf{Cylinder wake at $\mathrm{Re}=100$} (five seeds per cell; medians, range over seeds
in brackets). The field is the vorticity of a lattice-Boltzmann limit cycle, mean subtracted, so the
harmonics of the shedding frequency are physical rather than imposed. At both $\rho$ the second and
third harmonics fold past the Nyquist rate of $\Delta_1$.}
\label{tab:cylinder}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llccccc}
\toprule
$\rho$ & arm & $\mathcal{E}_{\Delta}$ & $\mathcal{E}_{1/2}$ &
$\mathcal{E}_{1/2}/\mathcal{E}_{\Delta}$ & $\kappa$ & $f_{\text{true}}$ \\
\midrule""")
    for w in (0.75, 0.763932):
        for i, (key, lab) in enumerate(arms):
            v = [r for r in rs if (r["model"], r["sampling"], r["dmd"]) == key and r["omega_mult"] == w]
            assert len(v) == 5
            pre = r"\multirow{5}{*}{" + RHO[w] + "}" if i == 0 else ""
            body = stats(v)
            if "lift" in lab: body = body.replace("$" + num(m([r["mid_over_grid"] for r in v])) + "$",
                                                  r"$\mathbf{" + num(m([r["mid_over_grid"] for r in v])) + "}$", 1)
            out.append(f"{pre} & {lab} & {body} \\\\")
        out.append(r"\midrule" if w == 0.75 else r"\bottomrule")
    out.append("\\end{tabular}%\n}\n\\end{table}")


if __name__ == "__main__":
    hdr = "% Bu dosya make_tables_v2.py tarafindan ham jsonl'den uretilir; elle duzenleme."
    dest = sys.argv[1] if len(sys.argv) > 1 else "."
    for fn, build in (("table_node", node_table), ("table_cyl", cyl_table)):
        out = [hdr]; build(out)
        p = os.path.join(dest, f"{fn}.tex")
        with open(p, "w") as f: f.write("\n".join(out) + "\n")
        print("yazildi:", p)

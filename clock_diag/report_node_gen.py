"""Table for the Neural ODE generalisation (HANDOFF 5-A), recomputed from the raw jsonl files.

Sources: kaggle_out_node_gen/results_node_gen_a.jsonl (18 runs) + kaggle_out_node_gen_b/
results_node_gen_b.jsonl (17) + results_node5.jsonl (5, the existing rho=1/4 regular arm).
40 runs, 5 seeds per cell, no duplicated configuration. The oracle column is the Koopman oracle
from results_merged2.jsonl at the same rho: there is no Neural ODE oracle, and the comparison the
paper needs is against an oracle-level inter-grid ratio, not against a specific architecture.

rho = omega_mult/2. Medians with [min,max] over seeds, because the spread across seeds is wide.
"""
import json, statistics as st

SRC = ["kaggle_out_node_gen/results_node_gen_a.jsonl",
       "kaggle_out_node_gen_b/results_node_gen_b.jsonl",
       "results_node5.jsonl"]


def load(p):
    with open(p) as f: return [json.loads(l) for l in f]


def main():
    rs = [r for p in SRC for r in load(p)]
    ks = [(r["model"], r["sampling"], r["omega_mult"], r["seed"]) for r in rs]
    assert len(set(ks)) == len(ks), "ayni konfigurasyon birden fazla kez var"
    cells = {}
    for r in rs: cells.setdefault((r["sampling"], r["omega_mult"]), []).append(r)
    m = st.median
    print(f"{'ornekleme':9s} {'rho':>6s} {'n':>2s} {'E_grid':>9s} {'E_mid':>9s} "
          f"{'mid/grid (medyan [min,max])':>28s} {'kappa':>5s} {'C':>5s} {'f_true':>6s}")
    for (s, w) in sorted(cells, key=lambda x: (x[0], x[1])):
        v = cells[(s, w)]; mg = [r["mid_over_grid"] for r in v]
        print(f"{s:9s} {w / 2:6.3f} {len(v):2d} {m([r['E_grid'] for r in v]):9.2e} "
              f"{m([r['E_mid'] for r in v]):9.2e} {m(mg):9.1f} [{min(mg):5.1f},{max(mg):6.1f}] "
              f"{m([r['kappa'] for r in v]):5.2f} {m([r['loop_closure'] for r in v]):5.2f} "
              f"{m([r['f_true'] for r in v]):6.3f}")
    orc = {}
    for r in load("results_merged2.jsonl"):
        if r["model"] == "oracle": orc.setdefault(r["omega_mult"], []).append(r)
    print("\nKoopman oracle (ayni rho, results_merged2.jsonl):")
    for w in sorted(orc):
        if w in {x[1] for x in cells}:
            print(f"  rho={w / 2:.3f} n={len(orc[w])} mid/grid={m([r['mid_over_grid'] for r in orc[w]]):8.2f}")


if __name__ == "__main__":
    main()

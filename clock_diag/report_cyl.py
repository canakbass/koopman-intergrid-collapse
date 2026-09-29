"""Table for the cylinder-flow benchmark (HANDOFF 5-B), recomputed from the raw jsonl.

Source: kaggle_out_cyl/results_cyl.jsonl -- 50 runs, five arms x 5 seeds x 2 rho, no failures.
Data is the Re=100 lattice-Boltzmann wake as a Fourier series in shedding phase, mean field
dropped (see data_cyl.py). rho = omega_mult/2, and at both values the second and third harmonics
of the shedding frequency fold past Nyquist.
"""
import json, statistics as st

SRC = "kaggle_out_cyl/results_cyl.jsonl"
ARMS = [(("koop", "regular", "none"), "olgu (duzenli ornekleme)"),
        (("koop", "mr", "none"), "kontrol (mr, lift yok)"),
        (("koop", "mr", "mr"), "lift, gate'siz"),
        (("koop", "mr", "mrg"), "lift, gate'li"),
        (("oracle", "regular", "none"), "oracle")]


def main():
    with open(SRC) as f: rs = [json.loads(l) for l in f]
    assert len(rs) == 50, f"beklenen 50 kayit, bulunan {len(rs)}"
    m = st.median
    for w in sorted({r["omega_mult"] for r in rs}):
        print(f"--- rho={w / 2:.3f} (omega_mult={w}) ---")
        print(f"{'kol':26s} {'n':>2s} {'E_grid':>9s} {'E_mid':>9s} "
              f"{'mid/grid medyan [min,max]':>27s} {'kappa':>6s} {'f_true':>6s}")
        for key, lab in ARMS:
            v = [r for r in rs if (r["model"], r["sampling"], r["dmd"]) == key and r["omega_mult"] == w]
            mg = [r["mid_over_grid"] for r in v]
            print(f"{lab:26s} {len(v):2d} {m([r['E_grid'] for r in v]):9.2e} "
                  f"{m([r['E_mid'] for r in v]):9.2e} {m(mg):8.2f} [{min(mg):6.2f},{max(mg):7.2f}] "
                  f"{m([r['kappa'] for r in v]):6.2f} {m([r['f_true'] for r in v]):6.3f}")
        # gate gercekten eliyor mu: mrg kolunda hangi modlar geciyor
        g = [r for r in rs if r["dmd"] == "mrg" and r["omega_mult"] == w]
        passed = [sum(r.get("gate_pass", [])) for r in g]
        print(f"  gate: gecen mod sayisi {passed}, gated_skip {[r.get('gated_skip') for r in g]}")
        print()


if __name__ == "__main__":
    main()

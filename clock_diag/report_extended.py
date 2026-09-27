# Ham-veri raporu: eski + yeni tohumlari birlestirip ortalama/aralik hesaplar.
# Sadece okur (results*.jsonl, kaggle_out*/results*.jsonl); hicbir kaynak dosyaya yazmaz.
import json, math, glob
from collections import defaultdict
import numpy as np
PI = math.pi

def load_all(patterns):
    seen, out = set(), []
    for pat in patterns:
        for fn in glob.glob(pat):
            for l in open(fn):
                l = l.strip()
                if not l or l in seen: continue
                seen.add(l)
                try: out.append(json.loads(l))
                except Exception: pass
    return out

def show(title, recs, key_fields, val_fields, seed_field="seed"):
    print(f"\n## {title}")
    G = defaultdict(list)
    for r in recs: G[tuple(r.get(k) for k in key_fields)].append(r)
    for k, rs in sorted(G.items(), key=lambda x: str(x[0])):
        rs = sorted(rs, key=lambda r: r.get(seed_field, 0))
        seeds = [r.get(seed_field) for r in rs]
        print(f"{dict(zip(key_fields,k))} n={len(rs)} seeds={seeds}")
        for vf in val_fields:
            vals = [r.get(vf) for r in rs if r.get(vf) is not None]
            if not vals: continue
            print(f"  {vf}: {['%.5g'%v for v in vals]}  mean={np.mean(vals):.5g}")

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("t2", "all"):
        # baseline: kaggle_out (s0) + moreseeds (s1-4); spectral rows: rerun (s0-1) + moreseeds (s2-4)
        base = [x for x in load_all(["kaggle_out/results.jsonl", "results_t2_moreseeds.jsonl"])
                if x.get("model") == "koop" and x.get("dmd", "none") == "none"]
        spec = [x for x in load_all(["kaggle_out_rerun/results_t2_rerun.jsonl"])] + \
               [x for x in load_all(["results_t2_moreseeds.jsonl"]) if x.get("seed", 0) >= 2]
        spec = [x for x in spec if x.get("model") == "koop" and x.get("dmd") in ("lock", "ft", "harm")]
        show("Table 2 (single-rate)", base + spec, ["dmd", "omega_mult"], ["mse_half", "mse_3_7", "f_true"])

    if which in ("t3", "all"):
        # control: kaggle_out_gpu (s0-2) + moreseeds (s3-4); treatment: rerun (s0-2) + moreseeds (s3-4)
        both = load_all(["kaggle_out_gpu/results.jsonl", "results_t3_moreseeds.jsonl"])
        ctl = [x for x in both if x.get("dmd", "none") == "none"]
        trt = load_all(["kaggle_out_rerun/results_t3_rerun.jsonl"]) + \
              [x for x in load_all(["results_t3_moreseeds.jsonl"]) if x.get("dmd") == "mr"]
        r = [x for x in ctl + trt if x.get("model") == "koop" and x.get("sampling") == "mr"]
        show("Table 3 (lifting)", r, ["dmd", "omega_mult"], ["mse_half", "mse_3_7", "f_true", "train_mse"])

    if which in ("threeobj", "all"):
        # per-mode lift of the three-object runs: principal frequency, lifted value, residual
        for x in sorted(load_all(["results3.jsonl"]), key=lambda x: (x.get("dmd", ""), x["seed"])):
            if x.get("dmd") != "mr": continue
            modes = [(abs(p), w, r) for p, w, r in zip(x["dmd_principal"], x["dmd_omegas"], x["mr_residual"])
                     if r is not None and abs(p) > 0.05]
            print(f"s{x['seed']} E1/2={x['mse_half']:.2e} " + "  ".join(
                f"p={p:.3f}->{w:+.2f}(r={r:.3f})" for p, w, r in sorted(modes)))

    if which in ("t5", "all"):
        r = load_all(["kaggle_out_partial/results_pend.jsonl", "kaggle_out_gate/results_pend.jsonl",
                       "kaggle_out_gpu/results_pend.jsonl", "results_pend_moreseeds.jsonl"])
        show("Table 5 (pendulum)", r, ["model", "dmd", "dmd_at", "sampling"], ["mse_half", "f_true", "partial_residual"])

    if which in ("axis1", "all"):
        r = load_all(["results_axis1.jsonl"])
        show("Axis1: n_osc=6 single object, sampling=mr", r, ["model", "dmd"], ["train_mse", "mse_half"])
        r3 = load_all(["kaggle_out_gpu/results.jsonl", "results_t3_moreseeds.jsonl"])
        r3 = [x for x in r3 if x.get("model") == "koop" and x.get("sampling") == "mr" and round(x.get("omega_mult", -1), 4) == 0.75]
        show("Axis1 ref: n_osc=4 single object, sampling=mr, omega=0.75", r3, ["dmd"], ["train_mse", "mse_half"])

    if which in ("axis2", "all"):
        r = load_all(["results_oneobj_wide.jsonl"])
        show("Axis2: n_osc=6,n_static=4 single object (data2 render)", r, ["model", "dmd"], ["train_mse", "mse_half"])
        r2 = load_all(["kaggle_out_scenario2/results2.jsonl", "results2_moreseeds.jsonl"])
        show("Axis2 ref: n_osc=6,n_static=4 two objects (Table 6)", r2, ["model", "dmd"], ["train_mse", "mse_half"])
        r3 = load_all(["results3.jsonl"])
        show("Three objects: n_osc=9,n_static=4", r3, ["model", "dmd"], ["train_mse", "mse_half"])

    if which in ("sweep", "all"):
        # alpha/beta sensitivity: E_1/2 and the locking margin of the lifted mode closest to the true frequency
        print("\n## alpha/beta sweep (rho_g, gate-free)")
        for x in sorted(load_all(["kaggle_out_sweep/sweep_results.jsonl"]), key=lambda x: (x["mr_frac"], x["dt2"], x["seed"])):
            t = x["omega_mult"] * PI
            i = min(range(len(x["dmd_omegas"])), key=lambda k: abs(abs(x["dmd_omegas"][k]) - t))
            print(f"alpha={x['mr_frac']:<7g} beta={x['dt2']:.3f} s{x['seed']}  E1/2={x['mse_half']:.1e}  "
                  f"margin(true)={x['mr_margin'][i]:.3f}")

    if which in ("node", "all"):
        r = load_all(["results.jsonl", "results_merged2.jsonl", "results_node5.jsonl"])
        r = [x for x in r if x.get("model") == "node" and round(x.get("omega_mult", -1), 4) == 0.5]
        show("Neural ODE rho=1/4", r, ["model"], ["kappa", "loop_closure", "chain_ratio", "mse_half"])

    if which in ("noise", "all"):
        r = load_all(["results_noise.jsonl"])
        show("Sprite noise sweep (dmd=mr, gate-free)", r, ["noise"], ["mse_half", "train_mse"])
        for rec in sorted(r, key=lambda x: (x.get("noise", 0), x.get("seed", 0))):
            print(" noise=%.2f seed=%s mr_residual=%s mr_margin=%s mse_half=%.5g" % (
                rec.get("noise"), rec.get("seed"), rec.get("mr_residual"), rec.get("mr_margin"), rec.get("mse_half", float("nan"))))

    if which in ("pendnoise", "all"):
        r = load_all(["results_pend_noise.jsonl"])
        for rec in sorted(r, key=lambda x: (x.get("noise", 0), x.get("seed", 0))):
            print(" noise=%.2f seed=%s gated_skip=%s partial_residual=%s mse_half=%.5g f_true=%s" % (
                rec.get("noise"), rec.get("seed"), rec.get("gated_skip"), rec.get("partial_residual"),
                rec.get("mse_half", float("nan")), rec.get("f_true")))

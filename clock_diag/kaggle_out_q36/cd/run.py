# Tek konfig: eğit -> teşhis -> results.jsonl + ckpt + faz portresi
import argparse, json, math, time, numpy as np, torch
import data, diagnostics, dmd_init
from models import KoopCT, NODE, KoopAmp
from losses import loss_fn, LAM

BAND = [0.125 * math.pi, 0.375 * math.pi, 0.625 * math.pi, 0.875 * math.pi]   # Nyquist bandına yayılmış, gerçek ω'yu içermez

def build(name, omega, seed, init="small"):
    if name == "koop": return KoopCT(seed=seed, omega_init=BAND if init == "band" else None)
    if name == "koopns":   # statik boyut yok: 6 osilatör, değişmezler genliklerde
        return KoopCT(n_osc=6, n_static=0, seed=seed, omega_init=[(i + 0.5) / 6 * math.pi for i in range(6)] if init == "band" else None)
    if name == "node": return NODE()
    if name == "koopamp": return KoopAmp(seed=seed)
    if name == "oracle":   # bir osilatör gerçek ω'da başlar (referans ölçek)
        g = torch.Generator().manual_seed(seed + 1000); w = (torch.randn(4, generator=g) * 0.1).tolist(); w[0] = omega
        return KoopCT(omega_init=w, seed=seed)
    raise ValueError(name)

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    omega = a.omega_mult * math.pi
    model = build(a.model, omega, a.seed, a.init).to(data.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed); t0 = time.time(); dmd_info = {}
    for it in range(a.iters):
        if a.dmd == "mrg" and it == a.dmd_at:    # eşikli çok hızlı DMD: yalnızca Δt2 artığı < tau olan modlar kilitlenir
            dmd_info = dmd_init.multirate_reinit(model, lambda n, K, gg, mode: data.batch(n, K, mode, omega, gg, noise=0.0),
                                                 torch.Generator().manual_seed(a.seed + 55), tau=a.tau)
            if not dmd_info.get("gated_skip"): opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif a.dmd == "mr" and it == a.dmd_at:     # çok hızlı DMD: Δt1 + Δt2 spektrumlarından tekil ω kaldırma
            dmd_info = dmd_init.multirate_reinit(model, lambda n, K, gg, mode: data.batch(n, K, mode, omega, gg, noise=0.0),
                                                 torch.Generator().manual_seed(a.seed + 55))
            model.omega.requires_grad_(False)
            opt = torch.optim.Adam([p for n, p in model.named_parameters() if n != "omega" and p.requires_grad], lr=1e-3)
        elif a.dmd != "none" and it == a.dmd_at:   # spektral yeniden başlatma: ızgara latentlerine DMD, L'yi eşitle
            dmd_info = dmd_init.reinit(model, lambda n, K, gg: data.batch(n, K, "regular", omega, gg, noise=0.0),
                                       torch.Generator().manual_seed(a.seed + 55), harmonic=a.dmd == "harm")
            if a.dmd in ("lock", "harm"): model.omega.requires_grad_(False)
            groups = [{"params": [p for n, p in model.named_parameters() if n != "omega" and p.requires_grad]}]
            if a.dmd == "ft": groups.append({"params": [model.omega], "lr": 1e-5})
            opt = torch.optim.Adam(groups, lr=1e-3)
        x, t = data.batch(64, 8, a.sampling, omega, g)
        loss, parts = loss_fn(model, x, t, a.variant, lam=0.0 if it < a.warmup else a.lam)
        opt.zero_grad(); loss.backward(); opt.step()
    if a.light:   # yalnızca ara zaman MSE + yörünge takibi (ağır teşhisler atlanır)
        model.eval(); diag = diagnostics.mse_grids(model, omega, "regular"); diag.update(diagnostics.track(diagnostics.rollout(model, omega), omega)); portrait = {}
    else:
        diag, portrait = diagnostics.full(model, omega, "regular" if a.sampling == "mr" else a.sampling)
    tag = f"{a.model}-{a.init}-dmd{a.dmd}-wu{a.warmup}_{a.variant}_w{a.omega_mult}_{a.sampling}_s{a.seed}"
    if portrait: torch.save(model.state_dict(), f"ckpt/{tag}.pt"); np.savez(f"ckpt/{tag}_portrait.npz", **portrait)
    rec = dict(model=a.model, init=a.init, dmd=a.dmd, dmd_at=a.dmd_at, **dmd_info, lam=a.lam, warmup=a.warmup, variant=a.variant, omega_mult=a.omega_mult, sampling=a.sampling, seed=a.seed,
               learned_omegas=model.omega.detach().cpu().tolist() if hasattr(model, "omega") else None,
               secs=round(time.time() - t0), last_parts=parts, **diag)
    with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="koop"); p.add_argument("--variant", default="base")
    p.add_argument("--omega_mult", type=float, default=0.5); p.add_argument("--sampling", default="regular")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--out", default="results.jsonl")
    p.add_argument("--lam", type=float, default=LAM); p.add_argument("--init", default="small"); p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--dmd", default="none", choices=["none", "lock", "ft", "harm", "mr", "mrg"]); p.add_argument("--dmd_at", type=int, default=500)
    p.add_argument("--light", action="store_true"); p.add_argument("--tau", type=float, default=0.1)
    main(p.parse_args())

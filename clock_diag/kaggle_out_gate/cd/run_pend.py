# Sarkaç kontrolü: aynı Koopman AE + (isteğe bağlı) DMD yeniden başlatma. Metrikler: train, Δ/2, 3Δ/7 MSE, ara zamanda açı takibi.
import argparse, json, math, time, numpy as np, torch, torch.nn.functional as F
import data, data_pend as DP, dmd_init
from models import KoopCT, KoopAmp

def main(a):
    torch.manual_seed(a.seed)
    Cls = KoopAmp if a.model.endswith("amp") else KoopCT
    if a.model.startswith("oracle"):
        g0 = torch.Generator().manual_seed(a.seed + 1000); w = (torch.randn(4, generator=g0) * 0.1).tolist(); w[0] = DP.W0
        model = Cls(omega_init=w, seed=a.seed)
    else: model = Cls(seed=a.seed)
    model.to(data.DEV); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed); info = {}; t0 = time.time()
    for it in range(a.iters):
        if a.dmd in ("partial", "partialg") and it == a.dmd_at:   # [1.5,4] rad/s bandında tek temel mod; partialg: artık ≥ tau ise iptal
            info = dmd_init.partial_install_info(model, lambda n, K, gg, mode: DP.batch(n, K, mode, gg, noise=0.0),
                                                 torch.Generator().manual_seed(a.seed + 55), tau=a.tau if a.dmd == "partialg" else None)
            if not info.get("gated_skip"): opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif a.dmd == "mr" and it == a.dmd_at:
            info = dmd_init.multirate_reinit(model, lambda n, K, gg, mode: DP.batch(n, K, mode, gg, noise=0.0), torch.Generator().manual_seed(a.seed + 55))
            model.omega.requires_grad_(False)
            opt = torch.optim.Adam([p for n, p in model.named_parameters() if n != "omega" and p.requires_grad], lr=1e-3)
        elif a.dmd != "none" and it == a.dmd_at:
            info = dmd_init.reinit(model, lambda n, K, gg: DP.batch(n, K, "regular", gg, noise=0.0), torch.Generator().manual_seed(a.seed + 55), harmonic=a.dmd == "harm")
            if a.dmd in ("lock", "harm"): model.omega.requires_grad_(False)
            groups = [{"params": [p for n, p in model.named_parameters() if n != "omega" and p.requires_grad]}]
            if a.dmd == "ft": groups.append({"params": [model.omega], "lr": 1e-5})
            opt = torch.optim.Adam(groups, lr=1e-3)
        x, t = DP.batch(64, 8, a.sampling, g)
        loss = F.mse_loss(model.predict(x[:, 0], t), x); opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); ge = torch.Generator().manual_seed(999); out = {}
    with torch.no_grad():
        th0 = DP.sample_th0(128, ge)
        for name, sp in [("train", 1.0), ("half", 0.5), ("3_7", 3 / 7)]:
            tt = torch.arange(0, 8 + 1e-6, sp, device=data.DEV)[None].repeat(128, 1)
            xt, _ = DP.video(th0, tt); out[f"mse_{name}"] = F.mse_loss(model.predict(xt[:, 0], tt), xt).item()
        tt = torch.arange(0, 8 + 1e-6, 1 / 16, device=data.DEV)[None].repeat(64, 1)
        xt, th = DP.video(th0[:64], tt); ang = DP.probe(model.predict(xt[:, 0], tt))
        fr = torch.remainder(tt, 1.0); mid = (fr >= 0.25) & (fr <= 0.75)
        out["f_true"] = ((ang - th).abs() < math.radians(8))[mid].float().mean().item()
    rec = dict(task="pendulum", model=a.model, dmd=a.dmd, dmd_at=a.dmd_at, sampling=a.sampling, seed=a.seed, **info, learned_omegas=model.omega.detach().cpu().tolist(),
               W0=DP.W0, secs=round(time.time() - t0), **out)
    with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="koop"); p.add_argument("--dmd", default="none", choices=["none", "lock", "ft", "harm", "mr", "partial", "partialg"])
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--sampling", default="regular")
    p.add_argument("--dmd_at", type=int, default=500); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=3000); p.add_argument("--out", default="results_pend.jsonl")
    main(p.parse_args())

# Ikinci senaryo (iki bagimsiz donen nesne): egit -> degerlendir -> results2.jsonl
# Ayni pipeline'in (silent failure + multi-rate lifting) tek-nesne renderer'ina ozgu olmadigini gosterir.
import argparse, json, math, time, numpy as np, torch, torch.nn.functional as F
import data2, dmd_init
from data import DEV
from models import KoopCT
from losses import loss_fn

N_OSC, N_STATIC = 6, 4  # d=16: iki gercek frekans icin ~3x yedek kapasite (tek-nesne probunda 4x idi)

def build(name, omega1, omega2, seed):
    if name == "oracle":
        g = torch.Generator().manual_seed(seed + 1000)
        w = (torch.randn(N_OSC, generator=g) * 0.1).tolist()
        w[0], w[1] = omega1, omega2
        return KoopCT(n_osc=N_OSC, n_static=N_STATIC, omega_init=w, seed=seed)
    return KoopCT(n_osc=N_OSC, n_static=N_STATIC, seed=seed)

@torch.no_grad()
def eval_mse(model, omega1, omega2, seed=999):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x, t = data2.batch2(128, 8, "regular", omega1, omega2, g, noise=0.0)
    out = {"train_mse": F.mse_loss(model.predict(x[:, 0], t), x).item()}
    sid1, th01, sid2, th02 = data2.test_set2(128, seed + 1)
    for name, sp in [("half", 0.5), ("3_7", 3 / 7)]:
        tt = torch.arange(0, 8 + 1e-6, sp, device=DEV)[None].expand(128, -1).contiguous()
        xt = data2.video2(sid1, th01, omega1, sid2, th02, omega2, tt)
        out[f"mse_{name}"] = F.mse_loss(model.predict(xt[:, 0], tt), xt).item()
    return out

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    omega1, omega2 = a.omega1_mult * math.pi, a.omega2_mult * math.pi
    model = build(a.model, omega1, omega2, a.seed).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed)
    dmd_info = {}
    for it in range(a.iters):
        if a.dmd == "mr" and it == a.dmd_at:
            dmd_info = dmd_init.multirate_reinit(
                model,
                lambda n, K, gg, mode: data2.batch2(n, K, mode, omega1, omega2, gg, noise=0.0),
                torch.Generator().manual_seed(a.seed + 55))
            model.omega.requires_grad_(False)
            opt = torch.optim.Adam([p for n, p in model.named_parameters() if n != "omega" and p.requires_grad], lr=1e-3)
        x, t = data2.batch2(64, 8, a.sampling, omega1, omega2, g)
        loss, parts = loss_fn(model, x, t, "base")
        opt.zero_grad(); loss.backward(); opt.step()
    diag = eval_mse(model, omega1, omega2)
    rec = dict(model=a.model, dmd=a.dmd, dmd_at=a.dmd_at, **dmd_info, omega1_mult=a.omega1_mult, omega2_mult=a.omega2_mult,
               sampling=a.sampling, seed=a.seed, learned_omegas=model.omega.detach().cpu().tolist(), **diag)
    with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="koop", choices=["koop", "oracle"])
    p.add_argument("--omega1_mult", type=float, default=0.75); p.add_argument("--omega2_mult", type=float, default=0.382)
    p.add_argument("--sampling", default="regular"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=3000); p.add_argument("--out", default="results2.jsonl")
    p.add_argument("--dmd", default="none", choices=["none", "mr"]); p.add_argument("--dmd_at", type=int, default=500)
    main(p.parse_args())

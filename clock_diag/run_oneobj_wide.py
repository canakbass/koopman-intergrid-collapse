# Gorev A, eksen 2 (nesne sayisi, sabit kapasite): run2.py'nin AYNI kapasitesi (n_osc=6,n_static=4,d=16)
# ama SADECE TEK hareketli nesne (data_oneobj_wide, data2'nin nesne-1 render'i). run2.py ile birebir
# ayni egitim/degerlendirme yapisi (mirror), sadece veri kaynagi tek-nesne. data.py/data2.py/run2.py/
# models.py/losses.py/dmd_init.py DEGISTIRILMEDI -- sadece import edildi.
import argparse, json, math, time, numpy as np, torch, torch.nn.functional as F
import data_oneobj_wide as D1, dmd_init
from data import DEV
from models import KoopCT
from losses import loss_fn

N_OSC, N_STATIC = 6, 4  # run2.py'deki iki-nesne kapasitesiyle birebir ayni (d=16)

def build(name, omega, seed):
    if name == "oracle":
        g = torch.Generator().manual_seed(seed + 1000)
        w = (torch.randn(N_OSC, generator=g) * 0.1).tolist()
        w[0] = omega
        return KoopCT(n_osc=N_OSC, n_static=N_STATIC, omega_init=w, seed=seed)
    return KoopCT(n_osc=N_OSC, n_static=N_STATIC, seed=seed)

@torch.no_grad()
def eval_mse(model, omega, seed=999):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    x, t = D1.batch1(128, 8, "regular", omega, g, noise=0.0)
    out = {"train_mse": F.mse_loss(model.predict(x[:, 0], t), x).item()}
    sid, th0 = D1.test_set1(128, seed + 1)
    for name, sp in [("half", 0.5), ("3_7", 3 / 7)]:
        tt = torch.arange(0, 8 + 1e-6, sp, device=DEV)[None].expand(128, -1).contiguous()
        xt = D1.video1(sid, th0, omega, tt)
        out[f"mse_{name}"] = F.mse_loss(model.predict(xt[:, 0], tt), xt).item()
    return out

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    omega = a.omega_mult * math.pi
    model = build(a.model, omega, a.seed).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed); t0 = time.time()
    dmd_info = {}
    for it in range(a.iters):
        if a.dmd == "mr" and it == a.dmd_at:
            dmd_info = dmd_init.multirate_reinit(
                model,
                lambda n, K, gg, mode: D1.batch1(n, K, mode, omega, gg, noise=0.0),
                torch.Generator().manual_seed(a.seed + 55))
            model.omega.requires_grad_(False)
            opt = torch.optim.Adam([p for n, p in model.named_parameters() if n != "omega" and p.requires_grad], lr=1e-3)
        x, t = D1.batch1(64, 8, a.sampling, omega, g)
        loss, parts = loss_fn(model, x, t, "base")
        opt.zero_grad(); loss.backward(); opt.step()
    diag = eval_mse(model, omega)
    rec = dict(model=a.model, dmd=a.dmd, dmd_at=a.dmd_at, **dmd_info, omega_mult=a.omega_mult,
               sampling=a.sampling, seed=a.seed, n_osc=N_OSC, n_static=N_STATIC, scenario="oneobj_wide",
               learned_omegas=model.omega.detach().cpu().tolist(), secs=round(time.time() - t0), **diag)
    with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="koop", choices=["koop", "oracle"])
    p.add_argument("--omega_mult", type=float, default=0.75)
    p.add_argument("--sampling", default="mr"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=3000); p.add_argument("--out", default="results_oneobj_wide.jsonl")
    p.add_argument("--dmd", default="none", choices=["none", "mr"]); p.add_argument("--dmd_at", type=int, default=500)
    main(p.parse_args())

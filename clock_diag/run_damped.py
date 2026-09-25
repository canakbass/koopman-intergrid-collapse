"""Görev A (bkz. NOTES_robustness.md): sönümlü/büyüyen (marjinal olmayan) özdeğer testi.
YENİ dosya. DEĞİŞTİRİLMEYEN dosyalar (sadece import): data.py (sample_times/DEV/H),
dmd_init.py (multirate_reinit -- AYNEN çağrılıyor). YENİ dosyalar: models_robustness.py
(KoopDamped), data_damped.py (render_damped/video_damped/batch_damped), bu dosya.

Hipotez: dmd_init.py'nin DMD/multi-rate makinesi özdeğerin sadece AÇISINI kullanıyor (modülüs
atılıyor) ve multirate_lift'teki atan2 ifadesi ölçek-bağımsız (kanıt: bkz. models_robustness.py
başlığı) -- bu yüzden dmd_init.py'ye HİÇ dokunmadan, sadece model sınıfına ayrı bir mu (sönüm)
parametresi ekleyerek, mevcut multi-rate kilitleme sönümlü sistemlerde de çalışmalı.

3 konfig x 3 tohum x 2 işaret (büyüyen/sönümlü) = 18 koşu:
  control    : dmd=none,  sampling=regular (hiç Δ2 görmüyor) -- ana makaledeki "sessiz başarısızlık"
               kontrolüyle AYNI tanım
  multirate  : dmd=mr,    sampling=mr (Δ1 + %12.5 Δ2), dmd_init.multirate_reinit AYNEN çağrılır
               -> model.omega kilitlenir, model.mu (DMD'siz) serbest gradyanla öğrenilmeye devam eder
  oracle     : dmd=none,  sampling=regular, model.omega[0] VE model.mu[0] gerçek değerle
               başlatılır (run.py'deki oracle mantığının mu'ya genişletilmesi), ikisi de serbest

Metrikler (inline; diagnostics.py KULLANILMIYOR -- data.py'nin sabit render'ına bağımlı, burada
gereksiz karmaşıklık): grid_mse (Δ1 ızgarası, eğitim dağılımı), E_half (t=k+0.5 SADECE ara-zaman
noktaları -- görülmemiş zamanlama, "sessiz başarısızlık" ölçütü).
"""
import argparse, json, math, time, numpy as np, torch, torch.nn.functional as F
import data, dmd_init
import data_damped as DD
from models_robustness import KoopDamped


def build(cfg, omega, mu, seed):
    if cfg in ("control", "multirate"):
        return KoopDamped(seed=seed)
    if cfg == "oracle":
        g = torch.Generator().manual_seed(seed + 1000)
        w = (torch.randn(4, generator=g) * 0.1).tolist(); w[0] = omega
        m = [0.0, 0.0, 0.0, 0.0]; m[0] = mu
        return KoopDamped(omega_init=w, mu_init=m, seed=seed)
    raise ValueError(cfg)


@torch.no_grad()
def eval_metrics(model, omega, mu, seed=999, n=128, K=8):
    g = torch.Generator().manual_seed(seed)
    sid = torch.randint(0, 3, (n,), generator=g).to(data.DEV)
    th0 = (torch.rand(n, generator=g) * 2 * math.pi).to(data.DEV)
    tt_grid = torch.arange(0, K + 1e-6, 1.0, device=data.DEV)[None].expand(n, -1).contiguous()
    tt_half = torch.arange(0, K + 1e-6, 0.5, device=data.DEV)[None].expand(n, -1).contiguous()
    x_grid = DD.video_damped(sid, th0, omega, mu, tt_grid)
    x_half = DD.video_damped(sid, th0, omega, mu, tt_half)
    grid_mse = F.mse_loss(model.predict(x_grid[:, 0], tt_grid), x_grid).item()
    xh_pred = model.predict(x_half[:, 0], tt_half)
    dense_mse = F.mse_loss(xh_pred, x_half).item()
    mask = torch.zeros(tt_half.shape[1], dtype=torch.bool)
    mask[1::2] = True                      # t = 0.5,1.5,...,7.5 -- SADECE ızgara-dışı noktalar
    e_half = F.mse_loss(xh_pred[:, mask], x_half[:, mask]).item()
    return dict(grid_mse=grid_mse, mse_half_dense=dense_mse, E_half=e_half)


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    omega = a.omega_mult * math.pi
    mu = a.mu
    model = build(a.cfg, omega, mu, a.seed).to(data.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time(); dmd_info = {}
    sampling = "mr" if a.cfg == "multirate" else "regular"
    for it in range(a.iters):
        if a.cfg == "multirate" and it == a.dmd_at:
            dmd_info = dmd_init.multirate_reinit(
                model, lambda n, K, gg, mode: DD.batch_damped(n, K, mode, omega, mu, gg, noise=0.0),
                torch.Generator().manual_seed(a.seed + 55))
            model.omega.requires_grad_(False)
            opt = torch.optim.Adam([p for n, p in model.named_parameters() if n != "omega" and p.requires_grad], lr=1e-3)
        x, t = DD.batch_damped(64, 8, sampling, omega, mu, g)
        pred = model.predict(x[:, 0], t)
        loss = F.mse_loss(pred, x)
        opt.zero_grad(); loss.backward(); opt.step()
        if not torch.isfinite(loss):
            break
    model.eval()
    metrics = eval_metrics(model, omega, mu, seed=999)
    rec = dict(task="damped", cfg=a.cfg, omega_mult=a.omega_mult, mu=mu, seed=a.seed,
               iters=a.iters, secs=round(time.time() - t0),
               learned_omegas=model.omega.detach().cpu().tolist(),
               learned_mu=model.mu.detach().cpu().tolist(),
               final_train_loss=float(loss.item()), **dmd_info, **metrics)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="control", choices=["control", "multirate", "oracle"])
    p.add_argument("--omega_mult", type=float, default=0.763932)
    p.add_argument("--mu", type=float, default=-0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--dmd_at", type=int, default=500)
    p.add_argument("--out", default="damped_results.jsonl")
    main(p.parse_args())

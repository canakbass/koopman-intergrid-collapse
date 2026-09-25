"""Görev D (bkz. NOTES_robustness.md): α (karışım oranı) ve β=Δt2/Δt1 duyarlılık taraması.
YENİ dosya. DEĞİŞTİRİLMEYEN dosyalar (sadece import): data.py, diagnostics.py (full -- AYNEN),
losses.py, dmd_init.py (dmd_modal / multirate_lift / _install -- AYNEN çağrılıyor, DEĞİŞTİRİLMEDEN).
YENİ: data_sweep.py (sample_times_sweep/batch_sweep, MR_FRAC/DT2'yi parametre yapıyor).

ÖNEMLİ İNCELİK: dmd_init.py::multirate_reinit, dmd_init.py::multirate_lift'i dt2 PARAMETRESİ
VERMEDEN çağırıyor (`multirate_lift(om, zp2)`), yani multirate_lift'in DEFAULT'u
(2**0.5/2) HER ZAMAN kullanılıyor -- β'yı süpürmek için multirate_reinit'i OLDUĞU GİBİ
kullanmak YANLIŞ olurdu (üretilen "d2" verisi gerçek dt2 ile aralıklı olsa bile, kaldırma adayı
eşleştirmesi sabit 0.7071 ile yapılır, β != 0.7071 iken YANLIŞ omega bulur). Bu yüzden burada
multirate_reinit'in mantığı ELLE, dt2'yi doğru şekilde multirate_lift'e ileterek tekrarlanıyor
(dmd_init.dmd_modal / dmd_init.multirate_lift / dmd_init._install -- HEPSİ DEĞİŞTİRİLMEDEN,
sadece dt2 argümanı AÇIKÇA geçiriliyor). Bu, dmd_init.py'ye dokunmadan mevcut, zaten
parametreli fonksiyonunu doğru kullanmaktır.
"""
import argparse, json, math, time, numpy as np, torch
import data, diagnostics, dmd_init
from run import build
from losses import loss_fn, LAM
import data_sweep as DS


def multirate_reinit_sweep(model, batch_fn, g, dt2, n_seq=256, K=8):
    with torch.no_grad():
        x1, _ = batch_fn(n_seq, K, g, "regular"); x2, _ = batch_fn(n_seq, K, g, "d2")
        enc = lambda x: model.enc(x.reshape(-1, 1, data.H, data.H)).reshape(n_seq, K + 1, -1).double().cpu().numpy()
        ze1, ze2 = enc(x1), enc(x2)
    P, om, info = dmd_init.dmd_modal(ze1, model.n, model.d - 2 * model.n)
    lifted, mi = dmd_init.multirate_lift(om, np.einsum("ij,ntj->nti", np.linalg.inv(P), ze2), dt2=dt2)
    info.update(mi); info["dmd_principal"] = [float(w) for w in om]
    info["dmd_omegas"] = [float(w) for w in lifted]
    with torch.no_grad():
        dmd_init._install(model, P, lifted)
    return info


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    omega = a.omega_mult * math.pi
    model = build("koop", omega, a.seed, a.init).to(data.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed); t0 = time.time(); dmd_info = {}
    for it in range(a.iters):
        if it == a.dmd_at:
            dmd_info = multirate_reinit_sweep(
                model, lambda n, K, gg, mode: DS.batch_sweep(n, K, mode, omega, gg, a.mr_frac, a.dt2, noise=0.0),
                torch.Generator().manual_seed(a.seed + 55), dt2=a.dt2)
            model.omega.requires_grad_(False)
            opt = torch.optim.Adam([p for n, p in model.named_parameters() if n != "omega" and p.requires_grad], lr=1e-3)
        x, t = DS.batch_sweep(64, 8, "mr", omega, g, a.mr_frac, a.dt2)
        loss, parts = loss_fn(model, x, t, a.variant, lam=0.0 if it < a.warmup else a.lam)
        opt.zero_grad(); loss.backward(); opt.step()
    diag, portrait = diagnostics.full(model, omega, "regular")
    rec = dict(model="koop", init=a.init, dmd="mr", dmd_at=a.dmd_at, mr_frac=a.mr_frac, dt2=a.dt2, **dmd_info,
               lam=a.lam, warmup=a.warmup, variant=a.variant, omega_mult=a.omega_mult, sampling="mr", seed=a.seed,
               learned_omegas=model.omega.detach().cpu().tolist(),
               secs=round(time.time() - t0), last_parts=parts, **diag)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="base")
    p.add_argument("--omega_mult", type=float, default=0.763932)
    p.add_argument("--mr_frac", type=float, default=data.MR_FRAC)
    p.add_argument("--dt2", type=float, default=data.DT2)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--dmd_at", type=int, default=500)
    p.add_argument("--out", default="sweep_results.jsonl")
    p.add_argument("--lam", type=float, default=LAM); p.add_argument("--init", default="small"); p.add_argument("--warmup", type=int, default=0)
    main(p.parse_args())

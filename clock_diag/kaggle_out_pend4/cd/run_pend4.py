"""FAZ 4: r-bagimli yerel sertifikanin (dmd_init_nonlinear.py::local_gate_residuals) IKINCI,
FARKLI bir genlik-bagimli-frekans nonlineer sistemde -- SERTLESEN (hardening) Duffing-tipi
acisal osilator, theta'' = -w0^2*theta - eps*theta^3 -- sinanmasi. Faz 3'un (run_pend3.py,
NOTES_nonlinear.md Bolum 7) full/restricted deney tasarimiyla BIREBIR ayni mantik; TEK fark
fizik: piksel sarkac (theta''=-w0^2*sin(theta), YUMUSAYAN, genlik arttikca frekans DUSER) yerine
Duffing (SERTLESEN, genlik arttikca frekans ARTAR) kullaniliyor. Amac: Faz 3'un negatif
sonucunun (6/6 tohumda negatif korelasyon, sicrama yok) sarkaca ozgu mu yoksa CNN-encoder
sertifikasyonunun genel bir sinirlamasi mi oldugunu ayirt etmek.

DEGISTIRILMEYEN dosyalar (sadece import): data.py, data_pend.py, data_pend2.py, models.py,
dmd_init_nonlinear.py, run_pend3.py. YENI dosyalar: data_duffing2.py (fizik), run_pend4.py
(bu dosya, egitim + degerlendirme).

Iki kol (Faz 3 ile AYNI genlik sinirlari, dogrudan karsilastirilabilir olmasi icin):
  full       : egitim genligi ~U(0.3,1.2) -- data_duffing2.sample_th0 (=data_pend.sample_th0) ile AYNI TAM aralik
  restricted : egitim genligi ~U(0.3,0.7) -- [0.7,1.2] egitimde HIC gorulmez

Her iki kol da ayni 256 degerlendirme yorungesiyle (SABIT tohum 777/1234+seed, TAM 0.3-1.2
araligindan, arm'dan BAGIMSIZ) degerlendirilir:
  (a) local_gate_residuals -- r-bagimli tutarlilik sertifikasi (DEGISMEDEN import edildi)
  (b) "gercek" frekans hatasi -- sifir-gecis (zero-crossing) tabanli gozlemlenen frekans
      (om_pred_obs, modelin KENDI dekodladigi cikistan) VS AYNI yontemle TRUE Duffing
      yorungesinden olculen om_true_obs. Faz 3'ten METODOLOJIK FARK: pendulum icin var olan
      kapali-form eliptik-integral frekans formulu (true_omega_pendulum) Duffing icin
      EZBERDEN yazilip GUVENILMEDI (gorev talimati geregi) -- onun yerine RK4 integrasyonunun
      kendisi enerji-korunumu kontroluyle DOGRULANDI (bkz. NOTES Bolum 8, kalibrasyon notu);
      "gercek hata" boylece om_pred_obs ile om_true_obs (ikisi de AYNI sifir-gecis yontemiyle,
      om_true_obs DOGRUDAN RK4 gercek yorungesinden) arasindaki fark olarak tanimlaniyor.
"""
import argparse, json, math, time, torch, torch.nn.functional as F
import data, data_pend as DP, data_duffing2 as DD, dmd_init_nonlinear as DI
from models import KoopAmp

W0, EPS = 2.4, 1.0   # kalibrasyon: bkz. NOTES_nonlinear.md Bolum 8 (eps=1.0 -> th0=0.3->1.2
                      # arasinda om ~%8.3 ARTIYOR, pendulumun ~%8-10 DUSMESIYLE karsilastirilabilir buyuklukte, TERS yonde)


def crossing_period(theta, t):
    """theta,t: ayni uzunlukta liste. Ardisik sifir-gecisleri (linear enterpolasyonla) arasi
    ortalama araligin 2 kati = donem T. <2 gecis varsa None (guvenilmez/basarisiz tahmin).
    run_pend3.py::crossing_period ile BIREBIR AYNI (kopyalandi, run_pend3.py degistirilmedi)."""
    crossings = []
    for i in range(len(theta) - 1):
        a, b = theta[i], theta[i + 1]
        if a == 0.0:
            crossings.append(t[i])
        elif (a < 0) != (b < 0):
            frac = a / (a - b)
            crossings.append(t[i] + frac * (t[i + 1] - t[i]))
    if len(crossings) < 2:
        return None
    diffs = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    return 2.0 * sum(diffs) / len(diffs)


def observed_omega(model, th0, t_fine):
    """th0 (B,), t_fine (T,) -> (om_true_obs list, om_pred_obs list), sifir-gecis yontemiyle.
    TRUE: DD.theta_at_duffing'in TAM RK4 yorungesi. PRED: modelin predict() ciktisi + DP.probe
    (probe ODE'den bagimsiz, aci-bankasi ayni -- bkz. data_duffing2.py basindaki not)."""
    B = th0.shape[0]
    tt = t_fine[None, :].expand(B, -1).to(data.DEV)
    x_true, th_true = DD.video_duffing(th0, tt, w0=W0, eps=EPS)
    with torch.no_grad():
        x_pred = model.predict(x_true[:, 0], tt)
        ang_pred = DP.probe(x_pred)
    tl = t_fine.tolist()
    th_true_l, ang_pred_l = th_true.tolist(), ang_pred.tolist()
    om_true_obs, om_pred_obs = [], []
    for i in range(B):
        Tt = crossing_period(th_true_l[i], tl)
        Tp = crossing_period(ang_pred_l[i], tl)
        om_true_obs.append(2 * math.pi / Tt if Tt else float("nan"))
        om_pred_obs.append(2 * math.pi / Tp if Tp else float("nan"))
    return om_true_obs, om_pred_obs


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = math.sqrt(sum((v - mx) ** 2 for v in xs))
    sy = math.sqrt(sum((v - my) ** 2 for v in ys))
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def main(a):
    torch.manual_seed(a.seed)
    model = KoopAmp(seed=a.seed).to(data.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed)
    lo, hi = (0.3, 1.2) if a.arm == "full" else (0.3, 0.7)
    t0 = time.time()
    diverged = False
    it = -1
    loss = torch.tensor(float("nan"))
    for it in range(a.iters):
        x, t = DD.batch_range(64, 8, a.sampling, g, lo, hi, w0=W0, eps=EPS)
        loss = F.mse_loss(model.predict(x[:, 0], t), x)
        opt.zero_grad(); loss.backward(); opt.step()
        if not torch.isfinite(loss):
            diverged = True
            break
    secs = round(time.time() - t0)
    model.eval()
    out = dict(task="pendulum4_duffing", arm=a.arm, seed=a.seed, lo=lo, hi=hi, w0=W0, eps=EPS,
               iters_done=it + 1, iters_target=a.iters, secs=secs, diverged=diverged,
               mse_train_final=float(loss.item()))
    if diverged:
        with open(a.out, "a") as f:
            f.write(json.dumps(out) + "\n")
        print(json.dumps(out))
        return
    with torch.no_grad():
        ge = torch.Generator().manual_seed(777)          # SABIT degerlendirme tohumu, arm'dan BAGIMSIZ
        th0_std = DD.sample_th0_range(256, ge, 0.3, 1.2)
        for name, sp in [("train", 1.0), ("half", 0.5), ("3_7", 3 / 7)]:
            tt = torch.arange(0, 8 + 1e-6, sp, device=data.DEV)[None].repeat(256, 1)
            xt, _ = DD.video_duffing(th0_std, tt, w0=W0, eps=EPS)
            out[f"mse_{name}"] = F.mse_loss(model.predict(xt[:, 0], tt), xt).item()

        # (a) r-bagimli sertifika: local_gate_residuals (dmd_init_nonlinear.py, DEGISTIRILMEDI)
        gc = torch.Generator().manual_seed(1234 + a.seed)
        th0_pair = DD.sample_th0_range(256, gc, 0.3, 1.2)   # sertifika-ozel orneklem, TUM aralik
        t1 = torch.arange(9, dtype=torch.float32)[None].expand(256, -1)
        t2 = torch.arange(9, dtype=torch.float32)[None].expand(256, -1) * (2 ** 0.5 / 2)
        x1, _ = DD.video_duffing(th0_pair, t1, w0=W0, eps=EPS)
        x2, _ = DD.video_duffing(th0_pair, t2, w0=W0, eps=EPS)
        # dmd_init_nonlinear.py::local_gate_residuals icinde `eye2 = torch.eye(2)[None]`
        # DEVICE BELIRTMEDEN olusturuluyor (her zaman CPU) -- bu dosyaya DOKUNULMUYOR (kural),
        # ama Kaggle GPU'da model/x1/x2 cuda oldugunda XtX = einsum(X,X) + 1e-6*eye2 satirinda
        # "cuda:0 ve cpu" cihaz uyusmazligi hatasi cikardi (6/6 is hata verdi, ilk Kaggle
        # denemesinde gozlendi). Cozum: protected dosyayi degistirmeden, sadece BU cagriyi
        # tamamen CPU'da yapiyoruz (modelin CPU kopyasi + CPU x1/x2) -- boylece fonksiyon
        # icindeki TUM tensorler (eye2 dahil) tutarli sekilde CPU'da kaliyor. Egitilmis
        # agirliklar deepcopy ile aynen tasiniyor, orijinal `model` (GPU) sonraki adimlar
        # (observed_omega) icin degismeden kaliyor.
        import copy
        model_cpu = copy.deepcopy(model).cpu().eval()
        r_enc, res, om_pred = DI.local_gate_residuals(model_cpu, x1.cpu(), x2.cpu())
        res_mean = res.mean(1)                              # (256,) 4 osilator ortalamasi

        # (b) gercek frekans hatasi -- sifir-gecis tabanli, om_pred_obs vs om_true_obs
        # (Duffing icin kapali-form formul YOK/GUVENILMEDI -- bkz. dosya basi ve NOTES Bolum 8)
        t_fine = torch.arange(0, 8 + 1e-6, 1 / 16)
        om_true_obs, om_pred_obs = observed_omega(model, th0_pair, t_fine)
        th0_abs_list = th0_pair.abs().tolist()
        true_err = [abs(p - tr) if (not math.isnan(p) and not math.isnan(tr)) else float("nan")
                    for p, tr in zip(om_pred_obs, om_true_obs)]

        edges = [0.3, 0.5, 0.7, 0.9, 1.1, 1.2]
        bins = []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            idx = [i for i, v in enumerate(th0_abs_list) if lo_e <= v < hi_e]
            if not idx:
                continue
            rr = [res_mean[i].item() for i in idx]
            ee = [true_err[i] for i in idx if not math.isnan(true_err[i])]
            bins.append(dict(lo=lo_e, hi=hi_e, n=len(idx),
                              mean_residual=sum(rr) / len(rr),
                              mean_true_err=(sum(ee) / len(ee)) if ee else None,
                              n_valid_freq=len(ee)))
        valid = [(res_mean[i].item(), true_err[i]) for i in range(256) if not math.isnan(true_err[i])]
        corr = pearson([v[0] for v in valid], [v[1] for v in valid])

        out["cert_bins"] = bins
        out["corr_residual_vs_true_err"] = corr
        out["n_valid_freq"] = len(valid)
        out["res_per_osc_mean"] = res.mean(0).tolist()
        out["raw"] = dict(th0_abs=th0_abs_list, residual_mean=res_mean.tolist(),
                           om_pred_obs=om_pred_obs, om_true_obs=om_true_obs, r_enc=r_enc.tolist())
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="full", choices=["full", "restricted"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=2200)
    p.add_argument("--sampling", default="mr")
    p.add_argument("--out", default="pend4_results.jsonl")
    main(p.parse_args())

"""Görev B (bkz. NOTES_robustness.md): "sadece birkaç ara-zaman etiketi eklesek olmaz mıydı?"
ablasyonu. YENİ dosya. DEĞİŞTİRİLMEYEN dosyalar (sadece import): data.py, diagnostics.py
(full/mse_grids -- AYNEN çağrılıyor, sonuçların results_merged2.jsonl / kaggle_out_gpu ile
DOĞRUDAN karşılaştırılabilir olması için), losses.py, run.py (sadece build() fonksiyonu import
ediliyor, model inşası BİREBİR AYNI). YENİ dosya: data_offgrid.py (sample_times_spot/batch_spot).

--sampling spot: eğitim döngüsü run.py'nin ana döngüsüyle AYNI, sadece data.batch yerine
data_offgrid.batch_spot çağrılıyor (dmd=none, model=koop, variant=base -- run.py'deki
"kontrol" konfigürasyonuyla BİREBİR AYNI, sadece zamanlama farklı). Değerlendirme HER ZAMAN
diagnostics.full(model, omega, "regular") ile yapılıyor (run.py'nin dmd=mr durumunda yaptığı
gibi -- eval her zaman aynı sabit ızgarada, training sampling'den bağımsız), böylece mse_half/
E_mid gibi alanlar results_merged2.jsonl (sampling=regular kontrol) ve kaggle_out_gpu
(sampling=mr, dmd=mr/none) satırlarıyla DOĞRUDAN karşılaştırılabilir.
"""
import argparse, json, math, time, numpy as np, torch
import data, diagnostics
from run import build
from losses import loss_fn, LAM
import data_offgrid as DO


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    omega = a.omega_mult * math.pi
    model = build(a.model, omega, a.seed, a.init).to(data.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed); t0 = time.time()
    for it in range(a.iters):
        if a.sampling == "spot":
            x, t = DO.batch_spot(64, 8, omega, g, frac=a.frac)
        else:
            x, t = data.batch(64, 8, a.sampling, omega, g)
        loss, parts = loss_fn(model, x, t, a.variant, lam=0.0 if it < a.warmup else a.lam)
        opt.zero_grad(); loss.backward(); opt.step()
    diag, portrait = diagnostics.full(model, omega, "regular")   # her zaman sabit ızgarada değerlendir (run.py'nin dmd=mr yolu gibi)
    tag = f"offgrid_{a.model}-{a.init}-{a.sampling}_frac{a.frac}_w{a.omega_mult}_s{a.seed}"
    rec = dict(model=a.model, init=a.init, dmd="none", lam=a.lam, warmup=a.warmup, variant=a.variant,
               omega_mult=a.omega_mult, sampling=a.sampling, frac=a.frac if a.sampling == "spot" else None,
               seed=a.seed, learned_omegas=model.omega.detach().cpu().tolist() if hasattr(model, "omega") else None,
               secs=round(time.time() - t0), last_parts=parts, **diag)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="koop"); p.add_argument("--variant", default="base")
    p.add_argument("--omega_mult", type=float, default=0.763932); p.add_argument("--sampling", default="spot")
    p.add_argument("--frac", type=float, default=data.MR_FRAC)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--out", default="offgrid_results.jsonl")
    p.add_argument("--lam", type=float, default=LAM); p.add_argument("--init", default="small"); p.add_argument("--warmup", type=int, default=0)
    main(p.parse_args())

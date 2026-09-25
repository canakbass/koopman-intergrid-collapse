"""Görev C (bkz. NOTES_robustness.md): decoder Lipschitz/düzgünlük kısıtı kontrolü.
YENİ dosya. DEĞİŞTİRİLMEYEN dosyalar (sadece import): data.py, diagnostics.py (full -- AYNEN
çağrılıyor, results_merged2.jsonl'daki kontrol satırlarıyla DOĞRUDAN karşılaştırılabilir
olması için), losses.py, run.py (sadece BAND sabiti import ediliyor). YENİ: models_robustness.py
(KoopCT_SNDec -- decoder'ın TÜM Linear/ConvTranspose2d katmanları spektral normalizasyon ile
sarılı, elle ayarlanmış bir ceza katsayısı YOK, standart mimari bir Lipschitz kısıtı).

Hipotez: Prop 1 -- decoder'ın serbest/esnek olması ızgara-dışı davranışı SERBEST bırakıyor
(E_AE küçük olması yeterlilik, düzgünlük değil). SN-decoder'lı control'ün E_1/2'si (mse_half)
SN'siz control'den (results_merged2.jsonl) daha mı düşük (kısıt tek başına yardımcı oluyor mu)
yoksa fark yok mu (kısıt tek başına yetmiyor, spektral tanılama/multi-rate gerçekten gerekli mi)?
Sadece dmd=none (control) konfigürasyonu -- multi-rate ile birleştirilmiyor, kısıtın TEK BAŞINA
etkisini izole ediyor.
"""
import argparse, json, math, time, numpy as np, torch
import data, diagnostics
from run import BAND
from losses import loss_fn, LAM
from models_robustness import KoopCT_SNDec


def build(seed, init):
    return KoopCT_SNDec(seed=seed, omega_init=BAND if init == "band" else None)


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    omega = a.omega_mult * math.pi
    model = build(a.seed, a.init).to(data.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(a.seed); t0 = time.time()
    for it in range(a.iters):
        x, t = data.batch(64, 8, "regular", omega, g)
        loss, parts = loss_fn(model, x, t, a.variant, lam=0.0 if it < a.warmup else a.lam)
        opt.zero_grad(); loss.backward(); opt.step()
    diag, portrait = diagnostics.full(model, omega, "regular")
    rec = dict(model="koop_sndec", init=a.init, dmd="none", lam=a.lam, warmup=a.warmup, variant=a.variant,
               omega_mult=a.omega_mult, sampling="regular", seed=a.seed,
               learned_omegas=model.omega.detach().cpu().tolist(),
               secs=round(time.time() - t0), last_parts=parts, **diag)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="base")
    p.add_argument("--omega_mult", type=float, default=0.763932)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--out", default="lipschitz_results.jsonl")
    p.add_argument("--lam", type=float, default=LAM); p.add_argument("--init", default="small"); p.add_argument("--warmup", type=int, default=0)
    main(p.parse_args())

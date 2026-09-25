"""r-bagimli sertifikanin (dmd_init_nonlinear.py) sentetik dogrulamasi.

Tasarim: iki KoopAmp modeli egit --
  full_range : r0 ~ U(0.3,1.2) -- TUM genlik araliginda goruyor (bkz. synth_amp_test.py treatment)
  restricted : r0 ~ U(0.3,0.7) -- SADECE alt yarida goruyor, [0.7,1.2] hic gorulmedi (ekstrapolasyon)

Sonra HER IKI modele, TUM r araliginda (0.3-1.2) yayilan degerlendirme yorungeleriyle
local_gate_residuals() uygulaniyor. Beklenti: full_range'de residual r boyunca kabaca
DUZ/DUSUK; restricted'te r<0.7'de dusuk, r>0.7'de (model hic gormedigi, g_j'nin
ekstrapole ettigi bolge) YUKSEK -- yani sertifika, modelin GUVENILIR OLMADIGI genlik
bolgesini DOGRU tespit ediyor mu, diye test ediliyor. Gercek K_TRUE bilgisiyle KARSILASTIRMA
YOK burada (residual sadece modelin KENDI iki-hizli tutarliligini olcuyor); dogruluk kontrolu
icin ayrica modelin gercek omega(r) hatasi (|om_pred - om_true|) da raporlaniyor, boylece
"residual yuksek <-> gercek hata da yuksek" korelasyonu dogrudan gosterilebiliyor.
"""
import json, math, torch
import synth_amp_test as S
import dmd_init_nonlinear as DI

def true_omega(r):
    return S.OMEGA0 * (1 + S.K_TRUE * r ** 2)

def eval_certificate(model, seed=777, n=256):
    g = torch.Generator().manual_seed(seed)
    x1, x2, r0_true = S.gen_pair(n, 8, g)     # r0_true ~ U(0.3,1.2) full degerlendirme araligi (egitimden BAGIMSIZ)
    r_enc, res, om_pred = DI.local_gate_residuals(model, x1, x2)
    res_j0, om_j0 = res[:, 0], om_pred[:, 0]
    true_err = (om_j0 - true_omega(r0_true)).abs()
    # ONEMLI: r_enc (modelin KENDI kodladigi yaricap) GENEL OLARAK r0_true ile AYNI OLCEKTE
    # DEGIL -- encoder, r'yi keyfi bir monoton donusumle yeniden-parametreleyebilir (bkz.
    # NOTES_nonlinear.md Bolum 3, ayni ozdeslenemezlik). Bu yuzden tablo GERCEK r0 ile
    # binleniyor (dogrulama amacli, gercekte r0 bilinmez ama burada biliniyor); ama
    # sertifikanin KENDISI (residual) sadece modelin r_enc/omega_eff'ine bakiyor, r0_true'ya
    # HIC erismiyor -- dogru sekilde "korlemesine" hesaplaniyor.
    edges = [0.3, 0.5, 0.7, 0.9, 1.1, 1.2]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r0_true >= lo) & (r0_true < hi)
        if m.sum() == 0: continue
        rows.append(dict(r0_lo=lo, r0_hi=hi, n=int(m.sum()),
                          mean_residual=float(res_j0[m].mean()),
                          mean_true_omega_err=float(true_err[m].mean())))
    corr = float(torch.corrcoef(torch.stack([res_j0, true_err]))[0, 1])
    return rows, corr

def run(iters=4000, seed=0):
    print("egitiliyor: full_range (r in 0.3-1.2, egitimde TUM aralik)...")
    m_full = S.train_koopamp_range(seed, iters, 0.3, 1.2)
    print("egitiliyor: restricted (r in 0.3-0.7, [0.7,1.2] HIC gorulmedi)...")
    m_restricted = S.train_koopamp_range(seed, iters, 0.3, 0.7)

    out = {}
    for name, m in [("full_range", m_full), ("restricted_0.3_0.7", m_restricted)]:
        rows, corr = eval_certificate(m)
        out[name] = dict(residual_by_r=rows, corr_residual_vs_true_error=corr)
        print(f"\n=== {name} === corr(residual, |om_pred-om_true|) = {corr:.3f}")
        for row in rows:
            print(f"  r0 in [{row['r0_lo']},{row['r0_hi']}) n={row['n']:3d}  residual={row['mean_residual']:.4f} rad  |om_err|={row['mean_true_omega_err']:.4f} rad/s")

    with open("synth_certificate_results.json", "w") as f:
        json.dump(out, f, indent=2)
    return out

if __name__ == "__main__":
    run()

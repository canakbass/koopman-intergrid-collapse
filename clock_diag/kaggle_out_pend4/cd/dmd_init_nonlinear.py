"""r-BAGIMLI sertifika: coklu-hizli tutarlilik testini TEK YORUNGE / TEK GENLIK duzeyine indirger.

AYRI arastirma dosyasi (bkz. NOTES_nonlinear.md Bolum 6). `dmd_init.py`'ye DOKUNULMADI,
sadece okundu; bu YENI bir dosya, mevcut kodu bozmuyor.

MOTIVASYON: `dmd_init.py`'deki multirate_reinit()/multirate_lift() TAMAMEN sabit-omega_j
varsayimiyla yazilmis -- butun veri havuzunda TEK bir DMD ile TEK bir omega_j kestirilip
tum yorungelere ayni sekilde uygulaniyor. KoopAmp'te ise omega_j ARTIK r_j'ye (yorungenin
kendi genligine) bagli; ama ONEMLI bir yapisal gercek hala geçerli: r_j = sqrt(a_j^2+b_j^2)
akis boyunca TAM KORUNUYOR (modelin omega_eff'i sadece dondurme uretiyor, r'yi degistirmiyor
-- bkz. models.py KoopAmp.omega_eff / synth_amp_test.py KoopCT._flow). Yani HER TEK yorunge
icin, o yorungenin r_j'sinde modelin ONGORDUGU omega_j(r_j) SABIT bir sayidir ve dmd_init.py
ile AYNI iki-hizli tutarlilik testi o TEK yorungeye dogrudan uygulanabilir:

  1) Yorungeyi Delta1 (egitim) izgarasinda kodla, t=0'daki (a_j,b_j)'den r_j ve modelin kendi
     omega_eff(a_j,b_j) = omega_j(r_j) tahminini al.
  2) AYNI baslangic kosulundan (ayni r0,ph0 -- fiziksel olarak ayni yorunge) Delta2 (ikinci,
     asal-olmayan oranli) izgarada kodla, ardisik latent orneklerden (dmd_init.py'deki
     multirate_lift ile AYNI en-kucuk-kareler dondurme-acisi kestirimi, ama TEK yorunge
     uzerinde) GOZLENEN faz artisini olc.
  3) residual_j = |tahmin edilen omega_j(r_j)*Delta2 - gozlenen faz| (2*pi/Delta1 tamsayi
     belirsizligi dmd_init.py'deki gibi cozulerek).

Bu residual GLOBAL bir "bu mod guvenilir mi" bayragi degil, YEREL bir "bu GENLIKTE model
kendi kendiyle tutarli mi" bayragi -- paper'in "gate" felsefesiyle (yerel dogrula, global
varsayma) birebir ayni, sadece eksen omega_j'den r_j'ye kayiyor. COK sayida yorungede
(farkli r0'lar) tekrarlanip residual(r) egrisine bakilirsa: model g_j'yi hangi genlik
araliginda dogru ogrenmisse orada residual dusuk, ogrenmediginde/ekstrapole ettiginde YUKSEK
olmasi beklenir -- bu asagida (synth_certificate_test.py) sentetik olarak DOGRULANIYOR.
"""
import math, torch

def local_gate_residuals(model, x1, x2, dt1=1.0, dt2=2 ** 0.5 / 2, kmax=4):
    """x1,x2: (B,T1,...)/(B,T2,...) AYNI (r0,ph0) baslangicindan iki izgarada gozlem
    (bkz. gen_pair helper'lari synth_amp_test.py / run_pend2.py'de).
    Donus: r (B,n_osc) [Delta1 izgarasinin t=0'indan], residual (B,n_osc) rad,
    om_pred (B,n_osc) [modelin omega_eff'inden]."""
    with torch.no_grad():
        B, T1 = x1.shape[0], x1.shape[1]
        T2 = x2.shape[1]
        z1 = model.enc(x1.reshape(B * T1, *x1.shape[2:])).reshape(B, T1, -1)
        z2 = model.enc(x2.reshape(B * T2, *x2.shape[2:])).reshape(B, T2, -1)
        n = model.n
        a0, b0 = z1[:, 0, 0:2 * n:2], z1[:, 0, 1:2 * n:2]
        r = (a0 ** 2 + b0 ** 2).sqrt()
        om_pred = model.omega_eff(a0, b0)                              # (B,n)
        res = torch.zeros(B, n)
        cd = lambda ang: (ang + math.pi) % (2 * math.pi) - math.pi
        ks = torch.arange(-kmax, kmax + 1, dtype=torch.float32)
        eye2 = torch.eye(2)[None]
        for j in range(n):
            X = z2[:, :-1, 2 * j:2 * j + 2]                            # (B,T2-1,2)
            Y = z2[:, 1:, 2 * j:2 * j + 2]
            XtX = torch.einsum('bti,btj->bij', X, X) + 1e-6 * eye2
            XtY = torch.einsum('bti,btj->bij', X, Y)
            Bm = torch.linalg.solve(XtX, XtY)                          # z_{t+1} ~ z_t @ Bm  (satir konvansiyonu)
            th2 = torch.atan2(Bm[:, 0, 1] - Bm[:, 1, 0], Bm[:, 0, 0] + Bm[:, 1, 1])
            cands = om_pred[:, j:j + 1] + 2 * math.pi * ks[None, :] / dt1     # (B,2kmax+1)
            dist = cd(cands * dt2 - th2[:, None]).abs()
            res[:, j] = dist.min(dim=1).values
    return r, res, om_pred

def bin_residual_by_r(r, res, osc_idx=0, edges=None):
    """r,res: (B,n_osc) -> r araliklarina gore ortalama residual (basit tanisal ozet)."""
    if edges is None: edges = [0.3, 0.5, 0.7, 0.9, 1.1, 1.2]
    rr, rv = r[:, osc_idx], res[:, osc_idx]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rr >= lo) & (rr < hi)
        out.append((lo, hi, int(m.sum()), float(rv[m].mean()) if m.any() else float('nan')))
    return out

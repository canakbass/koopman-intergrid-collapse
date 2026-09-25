"""Kontrollu sentetik test: genlige bagli frekans (KoopAmp tarzi r^2 modulasyonu) ogrenilebilir mi?

AYRI arastirma dosyasi -- paper/ ve paper_l4dc/ ile ilgisi yok, onlara yazilmadi/dokunulmadi.
Gercek piksel sarkactan (aksiyon-aci iliskisi eliptik integral, CNN encoder) kasitli olarak
AYRI tutuluyor: burada gercek uretici mekanizma tamamen bilinen, KAPALI FORMDA, POLINOM bir
omega(r^2) yasasiyla kuruluyor -- yani model sinifi (KoopAmp) ile veri-uretme sureci TAM
ESLESIYOR. Bu, yontem icin en ELVERISLI kosul: CNN piksel encoder karmasikligi yok, eliptik
integral yaklasiklama zorlugu yok. Eger yontem burada bile basarisiz olursa, bu sadece
"kucuk MLP transandantal fonksiyonu yakalayamiyor" degil, daha derin bir optimizasyon/
ozdeslenebilirlik sorunu oldugunu gosterir (bkz. NOTES_nonlinear.md).

Gercek dinamik (tam kapali form, sayisal entegrasyon YOK -- r korunumlu donme):
  r0 ~ U(0.3, 1.2)      (sarkac genlik araligiyla ayni, cf. data_pend.py)
  omega(r) = OMEGA0*(1 + K_TRUE*r^2)     -- pendulum'un kucuk-aci ACILIMININ kendisi
                                              (omega(A) ~ omega0*(1 - A^2/16) + ...), polinom.
  z(t) = r0*(cos(ph0+omega(r0)*t), sin(ph0+omega(r0)*t))

Gozlem: z, SABIT (egitilmeyen) rastgele bir MLP ile OBS_DIM'e gomuluyor -- CNN yerine
kontrollu bir "encoder'in ogrenmesi gereken nonlineer gozlem haritasi" rolu oynuyor.

Uc model karsilastiriliyor (hepsi ayni Enc/Dec MLP + KoopCT/KoopAmp mimarisini KULLANIYOR,
clock_diag/models.py ile ayni omega_eff formu, ama piksel degil bu sentetik gozlem icin):
  control    : KoopCT, tek sabit ogrenilebilir omega (genlik-korlugu taban çizgisi)
  treatment  : KoopAmp, g_j(r^2) sifirdan baslayip ogreniliyor (paper'daki basarisiz denemeyle AYNI mekanizma)
  true_law   : KoopAmp ama g_j DONDURULMUS, TAM DOGRU polinom yasaya esitlenmis (tavan/oracle)

Metrik: egitim izgarasinda (dt=1.0) MSE ve izgaralar-arasi (dt=1.0 offset 0.5) MSE (E_1/2 analogu).
"""
import math, json, torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
DEV = "cpu"

OBS_DIM = 10
OMEGA0 = 2.4
K_TRUE = -0.15          # omega(r^2) = OMEGA0*(1 + K_TRUE*r^2); pendulum kucuk-aci acilimiyla ayni isaret/mertebe
R_LO, R_HI = 0.3, 1.2    # data_pend.py ile ayni genlik araligi

# ---- sabit rastgele gozlem haritasi (OGRENILMIYOR; "gercek" gozlem sureci) ----
_go = torch.Generator().manual_seed(42)
_W1 = torch.randn(2, 48, generator=_go) * 0.9
_b1 = torch.randn(48, generator=_go) * 0.3
_W2 = torch.randn(48, OBS_DIM, generator=_go) * 0.6
_b2 = torch.randn(OBS_DIM, generator=_go) * 0.1

def true_obs(a, b):
    h = torch.tanh(a[..., None] * _W1[0] + b[..., None] * _W1[1] + _b1)
    return torch.tanh(h @ _W2 + _b2)

def sample_traj(B, K, dt, g, t_off=0.0, r_lo=R_LO, r_hi=R_HI):
    r0 = r_lo + (r_hi - r_lo) * torch.rand(B, generator=g)
    ph0 = 2 * math.pi * torch.rand(B, generator=g)
    t = t_off + torch.arange(K + 1, dtype=torch.float32) * dt
    om = OMEGA0 * (1 + K_TRUE * r0 ** 2)
    ph = ph0[:, None] + om[:, None] * t[None, :]
    a, b = r0[:, None] * torch.cos(ph), r0[:, None] * torch.sin(ph)
    x = true_obs(a, b)
    return x, t[None, :].expand(B, -1).clone(), r0

def gen_pair(B, K, g, dt1=1.0, dt2=2 ** 0.5 / 2, r_lo=R_LO, r_hi=R_HI):
    """AYNI (r0,ph0) baslangicindan iki farkli izgarada gozlem -- r-bagimli sertifika icin
    (bkz. dmd_init_nonlinear.py). Donus: x1 (B,K+1,OBS_DIM), x2 (B,K+1,OBS_DIM), r0 (B,)."""
    r0 = r_lo + (r_hi - r_lo) * torch.rand(B, generator=g)
    ph0 = 2 * math.pi * torch.rand(B, generator=g)
    om = OMEGA0 * (1 + K_TRUE * r0 ** 2)
    t1 = torch.arange(K + 1, dtype=torch.float32) * dt1
    t2 = torch.arange(K + 1, dtype=torch.float32) * dt2
    ph1 = ph0[:, None] + om[:, None] * t1[None, :]
    ph2 = ph0[:, None] + om[:, None] * t2[None, :]
    x1 = true_obs(r0[:, None] * torch.cos(ph1), r0[:, None] * torch.sin(ph1))
    x2 = true_obs(r0[:, None] * torch.cos(ph2), r0[:, None] * torch.sin(ph2))
    return x1, x2, r0

# ---- model: clock_diag/models.py ile ayni KoopCT/KoopAmp mantigi, piksel Enc/Dec yerine MLP ----
class Enc(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, d))
    def forward(s, x): return s.f(x)

class Dec(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, OBS_DIM))
    def forward(s, z): return s.f(z)

class KoopCT(nn.Module):
    def __init__(s, n_osc=1, n_static=2, omega_init=OMEGA0, seed=0):
        super().__init__()
        s.n, s.d = n_osc, 2 * n_osc + n_static
        s.enc, s.dec = Enc(s.d), Dec(s.d)
        s.omega = nn.Parameter(torch.tensor([float(omega_init)]))
    def omega_eff(s, a, b): return s.omega.expand(a.shape[0], -1)
    def _flow(s, z0, t):
        n = s.n; a, b, st = z0[:, 0:2*n:2], z0[:, 1:2*n:2], z0[:, 2*n:]
        ph = t[:, :, None] * s.omega_eff(a, b)[:, None, :]
        c, sn = torch.cos(ph), torch.sin(ph)
        osc = torch.stack([a[:, None] * c - b[:, None] * sn, a[:, None] * sn + b[:, None] * c], -1).flatten(-2)
        return torch.cat([osc, st[:, None].expand(-1, t.shape[1], -1)], -1)
    def latent(s, z0, t): return s._flow(z0, t)
    def predict(s, x0, t):
        z = s.latent(s.enc(x0), t); B, T, d = z.shape
        return s.dec(z.reshape(B * T, d)).reshape(B, T, OBS_DIM)

class KoopAmp(KoopCT):
    """treatment: g_j sifirdan ogreniliyor (paper'daki KoopAmp ile AYNI mekanizma)."""
    def __init__(s, **kw):
        super().__init__(**kw)
        s.g = nn.ModuleList([nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1)) for _ in range(s.n)])
        for m in s.g: nn.init.zeros_(m[2].weight); nn.init.zeros_(m[2].bias)
    def omega_eff(s, a, b):
        r2 = a ** 2 + b ** 2
        mod = torch.cat([s.g[j](r2[:, j:j + 1]) for j in range(s.n)], 1)
        return s.omega * (1 + mod)

class KoopAmpTrueLaw(KoopCT):
    """oracle/tavan: modulasyon DONDURULMUS, TAM DOGRU polinom yasaya esit (K_TRUE bilgisi verildi)."""
    def omega_eff(s, a, b):
        r2 = a ** 2 + b ** 2
        return s.omega * (1 + K_TRUE * r2[:, :s.n])

def train(model, iters, seed, K=8, dt=1.0, batch=64, lr=1e-3, r_lo=R_LO, r_hi=R_HI):
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    for it in range(iters):
        x, t, _ = sample_traj(batch, K, dt, g, r_lo=r_lo, r_hi=r_hi)
        loss = F.mse_loss(model.predict(x[:, 0], t), x)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def composite_loss(model, x, t, a1=1e-3, a2=1e-9, a3=1e-14):
    """Lusch, Kutz, Brunton 2018 (arXiv:1712.09707) Denklem 7a-7e, TAM katsayilarla (sarkac
    satiri, Tablo 3): alpha1=1e-3, alpha2=1e-9, alpha3=1e-14. L = a1*(Lrecon+Lpred) + Llin +
    a2*Linf + a3*||W||^2. Llin (latent-uzayda dogrudan lineerlik) agirlik=1 ile DOMINANT --
    bizim run_pend.py/synth_amp_test.py'nin basit tek-MSE kaybinda (~Lpred'e esdeger, agirlik
    1) HIC olmayan bir terim. S_p=30 yerine mevcut pencere uzunlugu (K adim) kullanildi."""
    B, T = x.shape[0], x.shape[1]
    x0 = x[:, 0]
    z0 = model.enc(x0)
    z = model.latent(z0, t)                                   # (B,T,d) analitik akis, K^m phi(x1) esdegeri
    if isinstance(z, tuple): z = z[0]
    ze = model.enc(x.reshape(B * T, -1)).reshape(B, T, -1)     # her karenin BAGIMSIZ kodlanmasi
    xr0 = model.dec(z0)
    Lrecon = F.mse_loss(xr0, x0)
    xhat = model.dec(z.reshape(B * T, -1)).reshape(B, T, -1)
    Lpred = F.mse_loss(xhat[:, 1:], x[:, 1:])
    Llin = F.mse_loss(ze[:, 1:], z[:, 1:])
    Linf = (xr0 - x0).abs().amax() + (xhat[:, 1] - x[:, 1]).abs().amax()
    l2 = sum((p ** 2).sum() for p in model.parameters())
    return a1 * (Lrecon + Lpred) + Llin + a2 * Linf + a3 * l2, dict(Lrecon=Lrecon.item(), Lpred=Lpred.item(), Llin=Llin.item(), Linf=Linf.item())

def train_rich(model, iters, seed, K=8, dt=1.0, batch=64, lr=1e-3, r_lo=R_LO, r_hi=R_HI, pretrain_frac=0.15):
    """Lusch ve ark. Destekleyici Bilgi'de belirtildigi gibi ("five minutes" otokodlayici-
    sadece on-egitim, tam kayip baslamadan once) -- once basit tahmin kaybiyla (Lrecon+Lpred
    esdegeri) isit, SONRA Llin agirlikli tam bilesik kayiba gec. pretrain_frac=0 -> on-egitim
    yok (ilk denemedeki gibi, karsilastirma icin)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    n_pre = int(iters * pretrain_frac)
    for it in range(iters):
        x, t, _ = sample_traj(batch, K, dt, g, r_lo=r_lo, r_hi=r_hi)
        if it < n_pre:
            loss = F.mse_loss(model.predict(x[:, 0], t), x)
        else:
            loss, _ = composite_loss(model, x, t)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def evaluate(model, seed=999, n=256, K=8, dt=1.0):
    g = torch.Generator().manual_seed(seed)
    x, t, r0 = sample_traj(n, K, dt, g)
    mse_train = F.mse_loss(model.predict(x[:, 0], t), x).item()
    # E_1/2 analogu: t=0'dan (r0,ph0) baslayip dt=0.5 izgarasinda (egitimde HIC gorulmeyen ofsetler) tahmin
    g2 = torch.Generator().manual_seed(seed)
    r0b = R_LO + (R_HI - R_LO) * torch.rand(n, generator=g2)
    ph0b = 2 * math.pi * torch.rand(n, generator=g2)
    omb = OMEGA0 * (1 + K_TRUE * r0b ** 2)
    tt = torch.arange(0, K + 1e-6, 0.5)
    ph = ph0b[:, None] + omb[:, None] * tt[None, :]
    ab, bb = r0b[:, None] * torch.cos(ph), r0b[:, None] * torch.sin(ph)
    xt = true_obs(ab, bb)
    x0 = true_obs(r0b * torch.cos(ph0b), r0b * torch.sin(ph0b))
    mse_half = F.mse_loss(model.predict(x0, tt[None].expand(n, -1)), xt).item()
    return dict(mse_train=mse_train, mse_half=mse_half, learned_omega=model.omega.detach().cpu().tolist())

def g_curve(model, rs):
    """Ogrenilen g_1(r^2) egrisini true K_TRUE*r^2 ile karsilastirmak icin ornekle."""
    if not hasattr(model, "g"): return None
    with torch.no_grad():
        r2 = torch.tensor(rs, dtype=torch.float32)[:, None]
        return model.g[0](r2).squeeze(-1).tolist()

def run(iters=4000, seeds=(0, 1, 2)):
    results = {}
    for name, Cls, kw in [("control", KoopCT, {}), ("treatment", KoopAmp, {}), ("true_law", KoopAmpTrueLaw, {})]:
        rows = []
        for sd in seeds:
            m = Cls(seed=sd, omega_init=OMEGA0)
            if name == "true_law": m.omega.requires_grad_(True)
            train(m, iters, seed=sd)
            m.eval()
            ev = evaluate(m, seed=999 + sd)
            if name == "treatment":
                rs = [0.3, 0.5, 0.7, 0.9, 1.1, 1.2]
                ev["g_curve_r"] = rs
                ev["g_curve_learned"] = g_curve(m, [r * r for r in rs])
                ev["g_curve_true"] = [K_TRUE * r * r for r in rs]
            rows.append(ev)
            print(name, sd, json.dumps(ev))
        results[name] = rows
    with open("synth_amp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

def run_rich(iters=4000, seeds=(0, 1, 2)):
    """Lusch-zengin kayipla KoopAmp (treatment_rich) -- basit-kayip treatment ile karsilastir.
    Sentetik ortamda basit kayip ZATEN oracle-seviyesindeydi (Bolum 3); burada zengin kaybin
    onu BOZMADIGINI (ayni iyi seviyede kaldigini) dogrulamak amacli hizli bir saglama."""
    rows = []
    for sd in seeds:
        m = KoopAmp(seed=sd, omega_init=OMEGA0)
        train_rich(m, iters, seed=sd)
        m.eval()
        ev = evaluate(m, seed=999 + sd)
        rows.append(ev)
        print("treatment_rich", sd, json.dumps(ev))
    with open("synth_amp_rich_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows

def train_koopamp_range(seed, iters, r_lo, r_hi, rich=False):
    m = KoopAmp(seed=seed, omega_init=OMEGA0)
    (train_rich if rich else train)(m, iters, seed=seed, r_lo=r_lo, r_hi=r_hi)
    m.eval()
    return m

if __name__ == "__main__":
    run()

# YENİ dosya (Görev A ve C, bkz. NOTES_robustness.md). models.py'ye HİÇ DOKUNULMADI, sadece
# import edilip genişletiliyor -- nonlineer hattın kendi kuralıyla (KoopAmp(KoopCT) gibi) aynı desen.
#
# --- Görev A: sönümlü/büyüyen (marjinal olmayan) özdeğer ---
# Gözlem (koordinatör tarafından önceden türetildi, bkz. görev talimatı): dmd_init.py::dmd_modal
# özdeğerin YALNIZCA açısını kullanıyor (`np.angle(lam[i])`); modülüs (`np.abs(lam)`) hesaplanıp
# `info['dmd_mod']`'a yazılıyor ama hiçbir yerde kullanılmıyor. multirate_lift'teki
# `th2 = atan2(A[1,0]-A[0,1], A[0,0]+A[1,1])` ifadesi de KANITLANABİLİR ölçek-bağımsız: gerçek
# bir sönümlü/büyüyen LTI modal çift için tek-adım haritası TAM OLARAK A = c*R(theta) biçimindedir
# (c = exp(mu*dt), theta = omega*dt -- rotasyon ve ölçekleme komütatif), bu yüzden
# atan2(2c*sin(theta), 2c*cos(theta)) = theta, c'den bağımsız. Yani dmd_init.py DEĞİŞTİRİLMEDEN
# sönümlü sistemlere uygulanabilir OLMALI -- yeter ki model sınıfı sönümü (mu_j) temsil edebilsin.
# KoopDamped bunu sağlıyor: ayrı bir mu parametresi, DMD ile KURULMUYOR (multirate_reinit sadece
# model.omega'yı kilitliyor / enc-dec'i P ile sarıyor), sadece gradyan inişiyle serbestçe öğreniliyor.
#
# --- Görev C: decoder Lipschitz/düzgünlük kısıtı ---
# Prop 1'in mantığı: decoder'ın serbest/esnek olması ızgara-dışı davranışı SERBEST bırakıyor
# (E_AE küçük olması yeterlilik değil). Standart, elle-ayarlanmış ceza katsayısı gerektirmeyen bir
# kısıt: decoder'ın Linear/ConvTranspose2d katmanlarını spektral normalizasyon (Lipschitz sabiti
# mimari olarak sınırlanır) ile sarmak.
import torch, torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from models import KoopCT


class KoopDamped(KoopCT):
    """ż = (L + diag(mu_j,mu_j) her osilatör çiftinde, statik kısımda 0)z.
    z_j(t) = exp(mu_j*t) * R(omega_j*t) * z_j(0) -- kapalı formda kalıyor (Prop 2'nin kapalı
    yörünge varsayımı burada YOK: mu_j != 0 iken yörünge artık kapanmıyor, spiral)."""

    def __init__(s, mu_init=None, **kw):
        super().__init__(**kw)
        m0 = torch.zeros(s.n) if mu_init is None else torch.tensor(mu_init, dtype=torch.float)
        s.mu = nn.Parameter(m0)

    def _flow(s, z0, t):
        n = s.n
        a, b, st = z0[:, 0:2 * n:2], z0[:, 1:2 * n:2], z0[:, 2 * n:]
        ph = t[:, :, None] * s.omega_eff(a, b)[:, None, :]
        env = torch.exp(t[:, :, None] * s.mu[None, None, :])
        c, sn = torch.cos(ph) * env, torch.sin(ph) * env
        osc = torch.stack([a[:, None] * c - b[:, None] * sn, a[:, None] * sn + b[:, None] * c], -1).flatten(-2)
        return torch.cat([osc, st[:, None].expand(-1, t.shape[1], -1)], -1)

    # latent() KoopCT'den DEĞİŞTİRİLMEDEN miras alınıyor -- s._flow çağrısı Python'un sanal
    # dispatch'i sayesinde otomatik olarak yukarıdaki override'a gidiyor, yeniden yazmaya gerek yok.

    def vel(s, z):
        # Otonom (t'den bağımsız) hız alanı: LTI sistem tanımı gereği ż sadece z'nin kendisine
        # bağlı olmalı. z=(a,b) noktasında: ȧ = mu*a - omega*b, ḃ = omega*a + mu*b (kompleks
        # gösterimde ż = (mu + i*omega)*z ile birebir aynı).
        n = s.n
        a, b = z[:, 0:2 * n:2], z[:, 1:2 * n:2]
        w = s.omega_eff(a, b)
        mu = s.mu[None, :]
        v = torch.stack([mu * a - w * b, w * a + mu * b], -1).flatten(-2)
        return torch.cat([v, torch.zeros_like(z[:, 2 * n:])], -1)


def make_sn_decoder(dec):
    """dec (models.Dec örneği) -> AYNI nesne, TÜM Linear/ConvTranspose2d katmanları spektral norm
    ile sarılmış (yerinde/in-place değiştirilir, yeni bir ağırlık kümesi YARATILMAZ -- init aynı
    kalır, sadece her forward'da ağırlık en büyük tekil değere bölünerek Lipschitz sabiti <= 1
    sağlanır katman başına)."""
    def wrap(seq):
        for i, m in enumerate(seq):
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d)):
                seq[i] = spectral_norm(m)
    wrap(dec.l)
    wrap(dec.c)
    return dec


class KoopCT_SNDec(KoopCT):
    """KoopCT ile AYNI (sönüm yok), tek fark: decoder'ın tüm Linear/ConvTranspose2d katmanları
    spektral normalizasyon ile sarılı -- decoder'ın Lipschitz sabiti mimari olarak sınırlı,
    elle ayarlanmış bir ceza katsayısı YOK."""

    def __init__(s, **kw):
        super().__init__(**kw)
        make_sn_decoder(s.dec)

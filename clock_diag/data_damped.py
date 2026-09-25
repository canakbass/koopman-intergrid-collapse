# YENİ dosya (Görev A, bkz. NOTES_robustness.md). data.py'ye DOKUNULMADI, sadece import edilip
# genişletildi: glyph artık sadece dönmüyor, aynı zamanda büyüyor/küçülüyor (skala = exp(mu*t)),
# modelin latent uzayındaki sönümlü/büyüyen bir modu temsil etmesi için sentetik gözlem.
import math, torch
import data
from data import DEV, H, GY, GX, _shapes


def render_damped(sid, ang, scale):
    """sid (N,), ang (N,), scale (N,) -> (N,1,H,H). data.render()'daki gibi ama u,v hesaplanmadan
    önce 1/scale ile ölçekleniyor: scale>1 => glyph ekranda BÜYÜR, scale<1 => KÜÇÜLÜR (glyph'in
    şekil fonksiyonu sabit yerel yarı-genişliklerle tanımlı; yerel koordinatı scale'e bölmek daha
    geniş bir ekran bölgesini aynı yerel şekle eşler)."""
    c, s = torch.cos(ang)[:, None, None], torch.sin(ang)[:, None, None]
    inv = (1.0 / scale)[:, None, None]
    u = (c * GX + s * GY) * inv
    v = (-s * GX + c * GY) * inv
    S = _shapes(u, v)
    return S.gather(0, sid[None, :, None, None].expand(1, -1, H, H))[0][:, None]


def video_damped(sid, th0, omega, mu, t):
    """sid (B,), th0 (B,), omega,mu float (skaler, sabit -- tek deney boyunca aynı), t (B,T)
    -> (B,T,1,H,H). ang = th0 + omega*t (dönme, data.video ile AYNI), scale = exp(mu*t)
    (büyüme/sönüm zarfı)."""
    B, T = t.shape
    ang = (th0[:, None] + omega * t).reshape(-1)
    scale = torch.exp(mu * t).reshape(-1)
    sid_r = sid.repeat_interleave(T)
    return render_damped(sid_r, ang, scale).reshape(B, T, 1, H, H)


def batch_damped(B, K, mode, omega, mu, g, noise=0.01):
    """data.batch ile AYNI arayüz (mode ∈ {regular,d2,mr,...} -> data.sample_times, DEĞİŞTİRİLMEDEN
    kullanılıyor), tek fark video_damped çağrısı (sabit omega,mu ile sönümlü/büyüyen gözlem)."""
    sid = torch.randint(0, 3, (B,), generator=g).to(DEV)
    th0 = (torch.rand(B, generator=g) * 2 * math.pi).to(DEV)
    t = data.sample_times(B, K, mode, g)
    x = video_damped(sid, th0, omega, mu, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t

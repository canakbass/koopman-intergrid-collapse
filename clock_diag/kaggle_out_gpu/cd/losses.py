# Varyantlar: base (sadece rollout rekonstrüksiyonu) | cons (+AE +Koopman latent tutarlılığı)
#             | cyc (+AE +ara zamanda manifold projeksiyonu enc(dec(z(s))) = z(s)) | both
import torch, torch.nn.functional as F
from data import H

LAM = 0.01   # kısıt ağırlığı: 1.0'da encoder sabite çöküyordu (AE kaybı 0.038, cons 1e-4)

def _var(z): return (z - z.mean(0, keepdim=True)).pow(2).mean() + 1e-6   # sabit kaydırmayla küçültülemez

def loss_fn(model, x, t, variant, lam=LAM):
    B, T = x.shape[:2]
    z0 = model.enc(x[:, 0])
    zp, zoff = model.latent(z0, t, offgrid=variant in ("cyc", "both"))
    rec = F.mse_loss(model.dec(zp.reshape(B * T, -1)).reshape(x.shape), x)
    parts = {"rec": rec.item()}; loss = rec
    if variant == "base": return loss, parts
    ze = model.enc(x.reshape(B * T, 1, H, H)).reshape(B, T, -1)
    if variant == "stat":   # statik boyutlar dizi boyunca sabit: θ0'ı statik boyutlarda ezberleme kanalını kapatır
        st = ze[..., 2 * model.n:]
        c = F.mse_loss(st[:, 1:], st[:, :1].expand_as(st[:, 1:])) / _var(st.reshape(B * T, -1).detach())
        parts["stat"] = c.item(); return loss + lam * c, parts
    ae = F.mse_loss(model.dec(ze.reshape(B * T, -1)).reshape(x.shape), x)
    loss = loss + ae; parts["ae"] = ae.item()
    if variant in ("cons", "both"):
        c = F.mse_loss(zp[:, 1:], ze[:, 1:]) / _var(ze.reshape(B * T, -1).detach())
        loss = loss + lam * c; parts["cons"] = c.item()
    if variant in ("cyc", "both") and zoff is not None:
        zs = zoff.reshape(-1, zoff.shape[-1])
        c = F.mse_loss(model.enc(model.dec(zs)), zs) / _var(zs.detach())
        loss = loss + lam * c; parts["cyc"] = c.item()
    return loss, parts

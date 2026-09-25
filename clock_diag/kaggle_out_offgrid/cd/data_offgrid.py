# YENİ dosya (Görev B, bkz. NOTES_robustness.md). data.py'ye DOKUNULMADI, sadece import edilip
# sample_times ŞABLON alınarak yeni bir "spot" modu eklendi: yöntemin (tutarlı bir ikinci Δ2
# örnekleme hızı) gerekliliğini test eden en ucuz kontrol -- dizilerin küçük bir kısmında
# (data.MR_FRAC ile AYNI oran, adil karşılaştırma), K adımdan SADECE BİRİ tam-sayı yerine
# ızgara-dışı, rastgele bir kesirli zamana kaydırılıyor (kalan adımlar hep Δ1 ızgarasında --
# yani "mr" modunun aksine TUTARLI bir ikinci hız YOK, sadece dağınık tek bir nokta).
import math, torch
import data
from data import DEV


def sample_times_spot(B, K, g, frac=data.MR_FRAC):
    """data.sample_times(mode='regular') ile AYNI taban ızgara (t=0..K, tamsayı); ek olarak
    dizilerin frac oranında, K adımdan rastgele seçilen BİRİNİN zamanı [k-1, k) aralığında
    U(0.1,0.9) ile ızgara-dışına kaydırılıyor (geri kalan adımlar tamsayı kalıyor)."""
    st = torch.ones(B, K)
    t = torch.cat([torch.zeros(B, 1), torch.cumsum(st, 1)], 1)   # [0,1,...,K], data.py 'regular' ile aynı
    rows = torch.rand(B, generator=g) < frac
    ridx = rows.nonzero(as_tuple=True)[0]
    if len(ridx) > 0:
        k = torch.randint(1, K + 1, (len(ridx),), generator=g)          # kaydırılacak adım (1..K)
        u = torch.rand(len(ridx), generator=g) * 0.8 + 0.1              # U(0.1,0.9)
        t[ridx, k] = (k - 1).to(torch.float32) + u
    return t.to(DEV)


def batch_spot(B, K, omega, g, frac=data.MR_FRAC, noise=0.01):
    """data.batch ile AYNI arayüz, tek fark sample_times_spot kullanması (render data.video,
    DEĞİŞTİRİLMEDEN çağrılıyor)."""
    sid = torch.randint(0, 3, (B,), generator=g).to(DEV)
    th0 = (torch.rand(B, generator=g) * 2 * math.pi).to(DEV)
    t = sample_times_spot(B, K, g, frac)
    x = data.video(sid, th0, omega, t)
    return x + noise * torch.randn(x.shape, generator=g).to(DEV), t

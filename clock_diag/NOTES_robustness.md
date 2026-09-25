# Hakem-gözü sağlamlık testleri: 4 önceden ele alınmamış nokta

Bu, `paper/` ve `paper_l4dc/`'den AYRI bir araştırma notu (o klasörlere hiç yazılmadı). Proje
sahibi makaleyi "hakem gözüyle" tekrar okuyup dört nokta buldu — hiçbiri daha önce test
edilmemişti. Aşağıda her biri için hipotez, ham sonuçlar (tohum bazında), ve dürüst bir
değerlendirme var. Tüm kod YENİ dosyalarda; `models.py, data.py, dmd_init.py, losses.py,
run.py, diagnostics.py` hiçbirine yazılmadı (mtime doğrulaması: aşağıda, Bölüm 5). Tüm koşular
Kaggle T4 GPU'da (anonymous Kaggle account), 3 tohum/konfig, `--iters 3000` (ana makalenin Table 1
protokolüyle aynı uzunluk).

**Önemli düzeltme (koordinatörün görev talimatındaki bir varsayımla ilgili):** Görev
talimatı Görev B ve D'de "mevcut `results_merged2.jsonl`'daki multi-rate (`sampling=mr`,
`dmd=mr`) satırları" diye bir referans veriyordu. Kontrol ettim: **`results_merged2.jsonl`
SADECE `sampling=regular` satırları içeriyor** (122/122 satır, doğrulama: `grep -o
'"sampling": "[a-z0-9_]*"' results_merged2.jsonl | sort | uniq -c`). Gerçek multi-rate
control/lock referans verisi `kaggle_out_gpu/results.jsonl`'da (`model=koop, sampling=mr,
dmd=none` VE `dmd=mr`, omega_mult ∈ {0.5, 0.75, 0.763932, 2.763932}, tohum 0-2). Aşağıdaki
karşılaştırmalarda BU dosyayı kullandım, `results_merged2.jsonl`'ı da (sampling=regular
kontrolü için) ayrıca kullandım. Bu ayrımı görmezden gelmek yanlış/yanıltıcı bir "multi-rate"
referansı olurdu.


## Görev A — Sönümlü/büyüyen (marjinal olmayan) özdeğer testi

**Hipotez (koordinatörün önceden türettiği, burada AMPİRİK olarak sınandı):** `dmd_init.py`
özdeğerin sadece açısını (`np.angle`) kullanıyor, modülüsü (`np.abs`) atıyor;
`multirate_lift`'teki atan2 ifadesi kanıtlanabilir ölçek-bağımsız (`M=c·R(θ)` ⇒ atan2 c'den
bağımsız θ verir). Gerçek bir sönümlü/büyüyen LTI modal çift TAM OLARAK bu formda davranıyor
(`z_{t+1} = e^{μΔt}R(ωΔt)z_t`), yani teori dmd_init.py'nin DEĞİŞTİRİLMEDEN sönümlü sistemlere
uygulanabileceğini söylüyor — yeter ki model sınıfı sönümü temsil edebilsin.

**Uygulama:** `models_robustness.py::KoopDamped(KoopCT)` — ayrı `mu` parametresi (n_osc
boyutlu, DMD ile KURULMUYOR, sadece gradyanla öğreniliyor), `_flow`/`vel` override (otonom LTI
alan: `ȧ=μa-ωb, ḃ=ωa+μb`). `data_damped.py` — glyph aynı zamanda `scale=exp(μt)` ile
büyüyüp/küçülüyor (render'ın `u,v`'si `1/scale` ile ölçekleniyor). μ=±0.08 seçildi (deneysel
doğrulama: `t∈[0,8]` boyunca `scale∈[0.53,1.90]`, piksel std/frac_on kontrolü ile glyph her
zaman görünür kaldığı teyit edildi — bkz. iş geçmişi). `run_damped.py` — 3 konfig
(control: dmd yok; multirate: `dmd_init.multirate_reinit` AYNEN çağrılıyor, `model.omega`
kilitleniyor, `model.mu` serbest; oracle: `omega[0]` VE `mu[0]` gerçek değerle başlatılmış,
ikisi de serbest), her ikisi işaret (büyüyen μ=+0.08 / sönümlü μ=-0.08) için 3 tohum = 18 koşu.
Metrikler inline: `grid_mse` (Δ1 ızgarası), `E_half` (SADECE t=k+0.5 noktaları — sıkı
ara-zaman/"sessiz başarısızlık" ölçütü).

### Ham sonuçlar (tohum bazında)

Kaynak: `kaggle_out_damped/damped_results.jsonl` (bkz. Bölüm 5 için tam yol). Kaggle: 18/18 iş OK, GPU
doğrulandı (`torch 2.10.0+cu128 cuda True`).

| μ | konfig | tohum 0 E_half | tohum 1 E_half | tohum 2 E_half | ortalama grid_mse |
|---|---|---|---|---|---|
| −0.08 (sönümlü) | control | 5.11e-02 | 5.24e-02 | 5.16e-02 | 2.40e-04 |
| −0.08 | multirate | 9.50e-05 | 1.25e-04 | 1.14e-04 | 1.15e-04 |
| −0.08 | oracle | 1.47e-04 | 1.47e-04 | 1.89e-04 | 1.19e-04 |
| +0.08 (büyüyen) | control | 1.69e-01 | 1.74e-01 | 1.77e-01 | 5.88e-04 |
| +0.08 | multirate | 2.30e-04 | 2.85e-04 | 2.62e-04 | 2.57e-04 |
| +0.08 | oracle | 2.94e-04 | 5.10e-04 | 4.16e-04 | 2.84e-04 |

Multi-rate kilitleme diagnostiği (gerçek osilatör çifti, |ω_true|=2.400 rad/s): tüm 6
multirate koşusunda doğru ω doğru bulundu (dmd_omegas ∈ [2.345, 2.465], gerçek hata < %3),
artık (`mr_residual`) 0.0006–0.011, marj (`mr_margin`) 0.63–0.76 — sönümsüz durumdakiyle
(kaggle_out_gpu, marj ~0.75) AYNI mertebede, sönümün kilitleme güvenini DÜŞÜRMEDİĞİ görülüyor.

### Değerlendirme: **DOĞRULANDI**

Her iki işarette de (sönümlü VE büyüyen) control, grid'de düşük hata verirken (2-6e-4) ara-
zamanda 170-700 kat daha kötü (E_half 0.05-0.18) — sessiz ara-zaman başarısızlığı sönümde de
AYNEN mevcut. Multi-rate kilitleme, dmd_init.py'ye HİÇ dokunmadan, E_half'ı oracle
mertebesine indiriyor (control'e göre 340-680× iyileşme, oracle'la aynı büyüklük sırasında,
hatta sönümlü kolda oracle'dan biraz daha iyi — muhtemelen oracle'ın μ[0] başlangıcının
gradyan inişiyle gerçek değerden uzaklaşması nedeniyle, bkz. ham veri: oracle'ın öğrenilen
μ'sü başlangıç -0.08'den -0.03/-0.05 civarına kaymış). Koordinatörün teorik türetmesi ampirik
olarak tam doğrulandı: mevcut DMD/multi-rate makinesi sönüme gerçekten kördür ve
değiştirilmeden sönümlü/büyüyen sistemlere genelliyor.


## Görev B — "Sadece birkaç ara-zaman etiketi eklesek olmaz mıydı?" ablasyonu

**Hipotez:** Dizilerin küçük bir kısmına (α=%12.5, mevcut MR_FRAC ile aynı oran) TEK bir
rastgele kesirli zaman noktası eklemek (tutarlı bir ikinci hız OLMADAN) yöntemin işini görür
mü, yoksa gerçekten tutarlı bir ikinci örnekleme hızı + spektral kaldırma mı gerekiyor?

**Uygulama:** `data_offgrid.py::sample_times_spot` — `data.sample_times`'ı taban alıp K
adımdan rastgele SEÇİLEN BİRİNİ `(k-1)+U(0.1,0.9)` kesirli zamanına kaydırıyor (geri kalanı
tam sayı). `run_offgrid.py` — `run.py::build` AYNEN import edilip kullanılıyor (model inşası
birebir aynı), değerlendirme HER ZAMAN `diagnostics.full(model, omega, "regular")` ile (run.py
dmd=mr durumunda yaptığı gibi), böylece sonuçlar mevcut satırlarla doğrudan karşılaştırılabilir.
α=%12.5 VE α=%25 (iki kat), omega_mult=0.763932, model=koop/init=small/variant=base/dmd=none,
3 tohum/α = 6 koşu.

### Ham sonuçlar (tohum bazında) — dört-yönlü karşılaştırma

| konfig | kaynak | tohum | mse_half | E_mid |
|---|---|---|---|---|
| control (sampling=regular, HİÇ off-grid yok) | results_merged2.jsonl | 2 | 0.04432 | 0.08705 |
| control | results_merged2.jsonl | 3 | 0.04339 | 0.08600 |
| control | results_merged2.jsonl | 4 | 0.04357 | 0.08452 |
| control | results_merged2.jsonl | 5 | 0.04290 | 0.08427 |
| spot α=0.125 (YENİ) | offgrid_results.jsonl | 0 | 0.02819 | 0.05659 |
| spot α=0.125 | offgrid_results.jsonl | 1 | 0.02744 | 0.05500 |
| spot α=0.125 | offgrid_results.jsonl | 2 | 0.03361 | 0.06124 |
| spot α=0.25 (YENİ) | offgrid_results.jsonl | 0 | 0.01834 | 0.04088 |
| spot α=0.25 | offgrid_results.jsonl | 1 | 0.01951 | 0.03638 |
| spot α=0.25 | offgrid_results.jsonl | 2 | 0.02488 | 0.04377 |
| mr sampling, dmd=none (tam Δ2 dizisi, KİLİTSİZ) | kaggle_out_gpu/results.jsonl | 0 | 0.02186 | — |
| mr, dmd=none | kaggle_out_gpu/results.jsonl | 1 | 0.02239 | — |
| mr, dmd=none | kaggle_out_gpu/results.jsonl | 2 | 0.02117 | — |
| mr + DMD kilit (YÖNTEM) | kaggle_out_gpu/results.jsonl | 0 | 0.0000961 | — |
| mr + DMD kilit | kaggle_out_gpu/results.jsonl | 1 | 0.0000637 | — |
| mr + DMD kilit | kaggle_out_gpu/results.jsonl | 2 | 0.0000830 | — |

(kaggle_out_gpu satırlarında E_mid alanı yok — o koşular farklı bir diagnostics.py sürümüyle/
`--light` olmadan ama eski alan setiyle kaydedilmiş; mse_half tüm satırlarda ortak, birincil
karşılaştırma metriği olarak kullanıldı.)

### Değerlendirme: **ÇÜRÜTÜLDÜ (yöntem gerçekten gerekli)**

Dağınık tek bir ara-zaman etiketi YARDIMCI OLUYOR ama YETERSİZ: α=%12.5 control'e göre mse_half'ı
~%35 azaltıyor (0.043→0.028), α=%25 ~%48 azaltıyor (0.043→0.022) — monoton ama KÜÇÜK bir
iyileşme. Buna karşılık gerçek yöntem (tutarlı Δ2 ızgarası + DMD kaldırma) mse_half'ı control'e
göre **~500× azaltıyor** (0.043→0.00008). Dahası, "mr sampling, dmd=none" satırı (modele TAM
bir Δ2 dizisi gösteriliyor ama DMD kilidi uygulanmıyor) bile spot-α=0.25'ten daha iyi değil
(0.022 vs 0.022, aynı mertebe) — yani sihir sadece "biraz daha fazla ızgara-dışı veri görmek"
değil, SPESİFİK OLARAK spektral kaldırma/kilitleme adımının kendisi. Bu, yöntemin gerekliliğini
güçlü şekilde destekliyor: ucuz "birkaç etiket ekle" alternatifi qualitatif farkı yaratmıyor.


## Görev C — Decoder düzgünlük/Lipschitz kısıtı kontrolü

**Hipotez:** Prop 1 — decoder'ın esnekliği ızgara-dışı davranışı serbest bırakıyor (E_AE
küçük olması yeterlilik, düzgünlük değil). Decoder'a mimari bir Lipschitz kısıtı (spektral
normalizasyon, elle ayarlanan ceza katsayısı YOK) eklenirse E_1/2 düşer mi?

**Uygulama:** `models_robustness.py::KoopCT_SNDec(KoopCT)` — decoder'ın TÜM Linear/
ConvTranspose2d katmanları `torch.nn.utils.parametrizations.spectral_norm` ile sarılı (yerinde,
init değişmeden). `run_lipschitz.py` — `run.py::BAND` import edilip aynı model inşa mantığı,
dmd=none (control), omega_mult=0.763932, 3 tohum.

### Ham sonuçlar (tohum bazında)

| konfig | tohum | mse_half | E_mid | E_grid |
|---|---|---|---|---|
| control (SN'siz, results_merged2.jsonl) | 2 | 0.04432 | 0.08705 | 0.000964 |
| control | 3 | 0.04339 | 0.08600 | 0.002833 |
| control | 4 | 0.04357 | 0.08452 | 0.000450 |
| control | 5 | 0.04290 | 0.08427 | 0.000801 |
| SN-decoder (YENİ) | 0 | 0.04612 | 0.09207 | 0.000416 |
| SN-decoder | 1 | 0.04913 | 0.09816 | 0.000583 |
| SN-decoder | 2 | 0.04596 | 0.09090 | 0.000702 |

### Değerlendirme: **ÇÜRÜTÜLDÜ (kısıt tek başına yardımcı olmuyor)**

SN-decoder control'ün E_1/2'si (mse_half ortalama 0.0471, E_mid ortalama 0.0937) SN'siz
control'den (mse_half ortalama 0.0436, E_mid ortalama 0.0855) **daha İYİ DEĞİL — hafifçe daha
KÖTÜ** (%8 daha yüksek mse_half). Grid hatası (E_grid) karşılaştırılabilir mertebede
(~4e-4 - 2.8e-3, iki grupta da), yani SN kısıtı eğitim-ızgarası performansını bozmuyor ama
ara-zaman genellemesini de İYİLEŞTİRMİYOR. Bu, Prop 1'in mantığını destekliyor: decoder'ı
mimari olarak "daha düzgün" yapmak (Lipschitz sabitini sınırlamak) başlı başına yeterli değil
— sorun decoder'ın DÜZGÜNLÜĞÜ değil, modelin latent'te DOĞRU FREKANSI bilmemesi; bu ancak
spektral tanılama/kilitleme (Görev B'nin de gösterdiği gibi) ile çözülüyor.


## Görev D — α (karışım oranı) ve β=Δt2/Δt1 duyarlılık taraması

**Hipotez:** β için mevcut TEORİK karakterizasyon (paper/sections/03_method.tex, Prop 1/
`prop:separation`) β'nın küçük-paydalı bir rasyonele yaklaştıkça çözünürlüğün bozulacağını
öngörüyor. Bu görev bunu AMPİRİK olarak sınıyor, artı α'ya (hiç taranmamış) bir duyarlılık
taraması ekliyor.

**Uygulama:** `data_sweep.py` — MR_FRAC/DT2 parametreli `sample_times_sweep`/`batch_sweep`.
`run_sweep.py` — **önemli incelik:** `dmd_init.multirate_reinit`, `multirate_lift`'i dt2
PARAMETRESİ VERMEDEN çağırıyor (her zaman varsayılan √2/2 kullanıyor) — β'yı süpürmek için bunu
OLDUĞU GİBİ kullanmak YANLIŞ olurdu (β≠0.7071 iken kaldırma yanlış dt2 ile eşleştirme yapardı).
Bunun yerine `multirate_reinit`'in mantığı `run_sweep.py::multirate_reinit_sweep` içinde ELLE
tekrarlandı, `dmd_init.dmd_modal`/`multirate_lift`/`_install` DEĞİŞTİRİLMEDEN ama dt2 AÇIKÇA
`multirate_lift(..., dt2=a.dt2)` ile geçirilerek. α taraması: β=√2/2 sabit, α∈{0.03,0.0625,0.25}
(α=0.125 referans zaten kaggle_out_gpu'da). β taraması: α=0.125 sabit, β∈{0.5 (kötü, 1/2
rasyonel), 0.51 (1/2'ye yakın), 0.618 (altın oran eşleniği, "iyi" irrasyonel), 0.85 ("iyi")}
(β=0.7071 referans zaten kaggle_out_gpu'da). omega_mult=0.763932, dmd=mr, 3 tohum/değer = 21 koşu.

### Ham sonuçlar — mse_half (tüm koşular başarılı, 21/21 OK)

**α taraması** (β=√2/2 sabit):

| α | tohum 0 | tohum 1 | tohum 2 | gerçek-osilatör marj (min-maks) |
|---|---|---|---|---|
| 0.03 | 1.64e-04 | 7.78e-05 | 1.51e-04 | 0.41–0.73 |
| 0.0625 | 1.35e-04 | 1.28e-04 | 7.94e-05 | 0.61–0.75 |
| 0.125 (referans, kaggle_out_gpu) | 9.61e-05 | 6.37e-05 | 8.30e-05 | ~0.66–0.76 |
| 0.25 | 7.79e-05 | 8.54e-05 | 8.37e-05 | 0.72–0.76 |

**β taraması** (α=0.125 sabit):

| β | tohum 0 | tohum 1 | tohum 2 | gerçek-osilatör marj (min-maks) |
|---|---|---|---|---|
| 0.5 (kötü, 1/2 rasyonel) | 7.26e-05 | 7.29e-05 | 8.45e-05 | **0.002–0.002** |
| 0.51 (1/2'ye yakın) | 7.78e-05 | 1.02e-04 | 8.83e-05 | 0.048–0.127 |
| 0.618 (altın eşlenik, iyi) | 8.86e-05 | 6.40e-05 | 8.15e-05 | 0.872–0.910 |
| 0.7071=√2/2 (referans, kaggle_out_gpu) | 9.61e-05 | 6.37e-05 | 8.30e-05 | ~0.66–0.76 |
| 0.85 (iyi) | 6.75e-05 | 5.56e-05 | 8.47e-05 | 0.812–0.905 |

### Değerlendirme: **KARIŞIK — teori marj metriğinde DOĞRULANDI, ama mse_half'ta beklenmedik şekilde sağlam**

**α:** mse_half tüm değerlerde aynı mertebede (7.8e-5–1.6e-4), α arttıkça monoton biraz
iyileşiyor; asıl fark GÜVEN MARJINDA görülüyor (α=0.03'te marj 0.41–0.73 iken α=0.25'te
0.72–0.76 — daha az veri = daha gürültülü/daha az kesin regresyon tahmini, beklenen istatistiksel
davranış). Yöntem α kadar düşük %3'te bile ÇALIŞIYOR — pratik açıdan cesaret verici bir bulgu
(makalede iddia edilmedi, burada keşfedildi).

**β:** Teori TAM OLARAK doğrulandı — ama SADECE marj metriğinde, mse_half'ta DEĞİL. β=0.5'te
(1/2 rasyonel) gerçek-osilatörün kilitleme marjı **0.002'ye çöküyor** (pratik anlamda sıfır —
aday frekanslar arasında neredeyse tam ikilik/çakışma; matematiksel neden: dt2=0.5 iken candidate
ω_k = ω0+2πk için θ_k=ω_k·0.5=ω0·0.5+πk, yani k'nin çift/tek sınıfları θ'yı sadece iki farklı
değere indirgiyor — Prop 1'in öngördüğü TAM OLARAK bu ambiguite). β=0.51 (1/2'ye yakın) marj
0.05–0.13'e kısmen toparlanıyor; β=0.618/0.85 ("iyi" irrasyonel-benzer) marj 0.81–0.91 ile
REFERANS β=0.7071'in marjından (0.66–0.76) bile YÜKSEK. **Ama** β=0.5'in düşük marjı mse_half'ı
BOZMADI (7.3e-5–8.5e-5, diğerleriyle aynı mertebe) — çünkü `dmd_init.py::multirate_lift`'teki
`+1e-3*np.abs(ks)` küçük-|k| tercih terimi, ambiguite sınıfı içinde HER ZAMAN en küçük |k|'lı
(gerçek, alias'sız) adayı seçiyor; yani nokta-tahmini (bu deneydeki gürültüsüz, sabit-tohum
değerlendirme koşulunda) korunuyor, ama GÜVEN MARJI teorinin öngördüğü gibi çöküyor — bu,
gürültü/dağılım kayması altında β=0.5'in kırılgan kalabileceğinin dolaylı ama net bir işareti
(bu deneyde DOĞRUDAN test edilmedi, aşırı-yorumlanmamalı). Sonuç: Prop 1'in β-çözünürlük
teorisi marj/güven ekseninde tam doğrulandı; mse_half ekseninde küçük-|k| ayrıştırma sezgiselinin
koruyucu etkisi nedeniyle β=0.5'te bile kırılma GÖZLENMEDİ — bu ikisi arasındaki ayrım dürüstçe
raporlanmalı, tek yönlü "doğrulandı" veya "çürütüldü" demek yanıltıcı olur.


## 5) Dosyalar ve doğrulama

**Yeni kod dosyaları** (hepsi `/home/can/Projeler/oneri_karsilastirma/clock_diag/` altında):
`models_robustness.py`, `data_damped.py`, `run_damped.py`, `build_damped_kernel.py`,
`data_offgrid.py`, `run_offgrid.py`, `build_offgrid_kernel.py`, `run_lipschitz.py`,
`build_lipschitz_kernel.py`, `data_sweep.py`, `run_sweep.py`, `build_sweep_kernel.py`,
bu dosya (`NOTES_robustness.md`).

**Ham sonuç JSONL dosyaları** (tohum-bazlı, koordinatörün kendi hesabı için):
- Görev A: `/home/can/Projeler/oneri_karsilastirma/clock_diag/kaggle_out_damped/damped_results.jsonl` (18 satır)
- Görev B: `/home/can/Projeler/oneri_karsilastirma/clock_diag/kaggle_out_offgrid/offgrid_results.jsonl` (6 satır) + referans `results_merged2.jsonl` (control) + `kaggle_out_gpu/results.jsonl` (multi-rate)
- Görev C: `/home/can/Projeler/oneri_karsilastirma/clock_diag/kaggle_out_lipschitz/lipschitz_results.jsonl` (3 satır) + referans `results_merged2.jsonl`
- Görev D: `/home/can/Projeler/oneri_karsilastirma/clock_diag/kaggle_out_sweep/sweep_results.jsonl` (21 satır) + referans `kaggle_out_gpu/results.jsonl` (α=0.125,β=0.7071 noktası)

Kaggle kernel günlükleri (tam stdout/stderr, hata ayıklama için): `kaggle_out_damped/
clock-damped.log`, `kaggle_out_offgrid/clock-offgrid.log`, `kaggle_out_lipschitz/
clock-lipschitz.log` (İLK deneme `dmd_init.py`'yi kernel'e paketlemeyi unuttuğu için 3/3 HATA
verdi — `ModuleNotFoundError: No module named 'dmd_init'`; `build_lipschitz_kernel.py`
düzeltilip YENİDEN push edildi, 3/3 OK oldu — şeffaflık için burada not ediliyor, ilk
denemenin logu da `kaggle_out_lipschitz/` içinde duruyor), `kaggle_out_sweep/clock-sweep.log`.

**Koruma doğrulaması:** `stat -c '%n %Y %s' models.py data.py dmd_init.py losses.py run.py
diagnostics.py` görev başında ve sonunda alındı, `diff` ile karşılaştırıldı — **HİÇBİR mtime/
boyut değişmedi**. `python3 dmd_init.py` görev başında ve sonunda çalıştırıldı — 4 selftest
(`selftest`, `selftest_lift`, `selftest_multirate`, `selftest_partial`) HER İKİSİNDE de OK.

**GitHub'a hiçbir şey push edilmedi. `paper/`, `paper_l4dc/`, `PROPOSED_addition.md`,
`PREREG.md`'ye dokunulmadı.**

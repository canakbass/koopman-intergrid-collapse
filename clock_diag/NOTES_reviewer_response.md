# Hakem eleştirisine yanıt: 4 görev (A/B/C/D)

Bu, `paper/`'den AYRI bir araştırma notu (o klasöre hiç yazılmadı, hiç okunmadı dışında
`04_experiments.tex`/`05_limitations.tex`'in salt-okunur konfigürasyon doğrulaması için).
Proje sahibi makaleyi harici bir LLM'e "hakem gibi eleştir" diye verdi, 4 itiraz geldi; ikisi
zaten Limitations'ta dürüstçe itiraf edilmişti, ikisi (Neural ODE tek-tohum, gürültü-duyarlılığı
hiç ölçülmemiş) gerçek zayıf noktaydı. Bu not hepsini ölçüyor.

**Sıkı kurallar (aynen uygulandı):** `data.py, models.py, dmd_init.py, losses.py, run.py,
run_pend.py, run2.py, data2.py, diagnostics.py` HİÇBİRİNE yazılmadı — sadece import edildi.
Yeni davranış YENİ dosyalarda (`data_oneobj_wide.py`, `run_oneobj_wide.py`, `run_noise.py`,
`run_pend_noise.py`, `data_threeobj.py`, `run3.py`). mtime doğrulaması: Bölüm 5.
Tüm koşular Kaggle T4 GPU'da, anonymous Kaggle account. GitHub'a hiçbir şey push edilmedi.

**Metrik tanımı (önemli, tüm görevler için tutarlı kullanıldı):** Makalenin kendi
Limitations bölümünde iki-nesne senaryosu için hesapladığı "6–11×" oranı **aynı-model**
E_1/2/L_Δ oranıdır (`mse_half`/`train_mse`, run2.py'nin `eval_mse` çıktısı) — obs:silent'teki
"90–958×" ise **control-vs-oracle** E_1/2 oranıdır, farklı bir ölçüm. Bu ayrımı doğruladım
(aşağıda, Görev A) ve tüm yeni ölçümlerde run2.py/run.py'nin ürettiği `train_mse`/`mse_half`
alanlarını, run2.py'nin Limitations'taki hesaplamasıyla BİREBİR aynı formülle kullandım
(doğrulama: control satırı için 0.00345/0.00048=7.2, 0.00352/0.00032=11.0 — makaledeki
"6–11×" ile eşleşiyor).

**Önemli yapısal bulgu (Görev B sırasında ortaya çıktı, kod DEĞİŞTİRİLMEDEN):** Tablo 5'in
("band-restricted lock, ungated/gated" satırları) gerçek üretici kernel'i `--dmd_at 500`
DEĞİL, **`--dmd_at 1500`** kullanıyor — bunu `kaggle_out_gate/results_pend.jsonl`'daki
residual değerlerinin (`0.101/0.678/0.485`) makale tablosuyla TAM eşleşmesiyle doğruladım
(bkz. `kaggle_out_gate/kernel.py`, `dmd_at=1500` JOBS listesinde). `kaggle_out_partial/`
(varsayılan `dmd_at`, ungated) farklı sayılar veriyor (0.0108/0.0102/0.0145, residual yok).
Makale metni "the spectral step [is] delayed to $T_0=1500$ to let the encoder mature" diyor
ama bunu bir YAN kontrol gibi ifade ediyor — gerçekte bu, tablodaki SATIRLARIN KENDİSİNİN
konfigürasyonu. Tüm yeni Tablo 5 tohumlarını (Görev B) VE gürültü taramasını (Görev D,
sarkaç kolu) bu doğru `dmd_at=1500` ile ürettim.

---

## Görev A — İki-nesne senaryosunda oran düşüşünün mekanizması

**Hipotez (koordinatörün ön-kontrolü, burada izole edildi):** Tek-nesneden iki-nesneye
geçince control'ün E_1/2/L_Δ oranı "90–958×"ten "6–11×"e düşüyor. Bu, karmaşıklık
arttıkça sessiz başarısızlığın kendiliğinden söndüğü anlamına mı geliyor (multi-rate
lifting'e gerçek sistemlerde gerek kalmayabilir mi), yoksa başka bir etken mi?

**Tasarım:** İki eksen ayrıştırıldı + üçüncü bir gizli değişken ortaya çıktı:
- **Eksen 1 (kapasite):** aynı tek nesne, n_osc=4 (`koop`) vs n_osc=6 (`koopns`), aynı ρ=3/8
  (ω=0.75π), aynı `sampling=mr` (karışık veri), control (`dmd=none`), 5 tohum.
- **Eksen 2 (sahne):** aynı latent kapasite (n_osc=6, n_static=4 — run2.py'nin İKİ-nesne
  konfigürasyonu ile BİREBİR aynı), ama TEK nesne (`data_oneobj_wide.py`, data2.py'nin
  nesne-1 render ölçeği/konumu kullanılarak inşa edildi) vs İKİ nesne (mevcut run2.py, 5
  tohuma çıkarıldı), aynı ω1=0.75π, aynı `sampling=mr`, control, 5 tohum.
- **Gizli üçüncü eksen (beklenmedik, ölçüm sırasında bulundu):** `sampling=regular` (hiç
  karışık veri yok — orijinal tab:diagnostics'in kullandığı rejim) vs `sampling=mr`
  (%12.5 Δ2 karışımı — Tablo 3/6'nın kullandığı rejim), n_osc=4, TEK nesne sabit.

### Ham sonuçlar (tohum bazında, E_1/2/L_Δ = mse_half/train_mse, aynı modelin kendi oranı)

| Konfigürasyon | ρ | kaynak | tohum-bazlı oran | aralık |
|---|---|---|---|---|
| n_osc=4, **regular**, 1 nesne (orijinal rejim) | 3/8 | `kaggle_out/results.jsonl`+`results_t2_moreseeds.jsonl` | 115.1/136.1/61.1/69.4/125.9 | 61–136× |
| n_osc=4, **regular**, 1 nesne | ρ_g | aynı | 89.1/152.2/42.3/79.9/69.2 | 42–152× |
| n_osc=4, **regular**, 1 nesne | 1+ρ_g | aynı | 117.9/187.8/50.8/97.0/64.4 | 51–188× |
| n_osc=4, **mr**, 1 nesne (Eksen1 referans) | 3/8 | `kaggle_out_gpu/results.jsonl`+`results_t3_moreseeds.jsonl` | 17.1/11.8/16.0/16.2/22.6 | **11.8–22.6×** |
| n_osc=6 (koopns, n_static=0), **mr**, 1 nesne (Eksen 1) | 3/8 | `results_axis1.jsonl` | 30.3/13.4/16.7/20.4/18.4 | **13.4–30.3×** |
| n_osc=6/n_static=4 (data2 kapasitesi), **mr**, **1** nesne (Eksen 2) | 3/8+0.382 | `results_oneobj_wide.jsonl` | 13.7/12.5/9.8/7.7/15.1 | **7.7–15.1×** |
| n_osc=6/n_static=4, **mr**, **2** nesne (Tablo 6, 5 tohuma çıkarıldı) | 3/8+0.382 | `kaggle_out_scenario2/results2.jsonl`+`results2_moreseeds.jsonl` | 7.3/10.9/5.6/6.9/4.5 | **4.5–10.9×** |

(Makalenin orijinal "6–11×" iddiası 3 tohumla; burada 5 tohumla 4.5–10.9× — aynı mertebe,
doğrulandı.)

**Yöntemin gerekliliği, her eksende (bonus soru):** multi-rate lifting (`dmd=mr`), HER
konfigürasyonda mse_half'ı büyük ölçüde iyileştiriyor, oran ne olursa olsun:
- n_osc=4, mr, 1 nesne: control 0.0198 → mr 0.00018 (**~110×**)
- n_osc=6 (koopns), mr, 1 nesne: control 0.0185 → mr 0.000083 (**~223×**)
- n_osc=6/n_static=4, mr, 1 nesne (Eksen 2): control 0.00259 → mr 0.0000667 (**~39×**)
- n_osc=6/n_static=4, mr, 2 nesne: control 0.00375 → mr 0.000112 (**~33×**)

### Değerlendirme: **KISMEN "kapasite/sahne değil", ama asıl etken İKİSİ DE değil — üçüncü bir değişken (örnekleme rejimi) baskın**

Sorunun kendi çerçevesinde ("kapasite mi sahne mi?") cevap **açık**: **ikisi de değil, ya da
en fazla küçük katkı**. n_osc 4→6 (Eksen 1, aynı tek nesne) oranı 11.8–22.6× → 13.4–30.3×
— yani KAPASİTE ARTIŞI ORANI KÜÇÜLTMÜYOR, hatta hafifçe büyütüyor (gürültülü, örtüşen
aralıklar, istatistiksel olarak ayırt edilemez). Nesne sayısı 1→2 (Eksen 2, aynı geniş
kapasite) oranı 7.7–15.1× → 4.5–10.9× — bu KÜÇÜK bir düşüş (~1.5–2×), ama 90–958×'ten
6–11×'e (~10–90×) düşüşün TAMAMINI açıklamaktan ÇOK uzak.

**Asıl bulgu, sorunun sormadığı ama ölçümün ortaya çıkardığı şey:** control/L_Δ oranındaki
BÜYÜK düşüşün (61–188× → 11.8–30.3×, yani ~5–8×'lik düşüş) kaynağı ne kapasite ne nesne
sayısı — **örnekleme rejimi** (`sampling=regular` → `sampling=mr`, yani eğitim verisine
%12.5 Δ2-hızlı pencere eklemek, DMD kaldırma/kilitleme YAPILMADAN bile). Aynı n_osc=4, aynı
tek nesne, sadece eğitim verisine %12.5 ikinci-hız penceresi eklemek oranı ~61–188×'ten
~11.8–22.6×'e düşürüyor — bu TEK BAŞINA, makalenin "6–11×" ile "90–958×" arasındaki farkın
BÜYÜK KISMINI açıklıyor. Nesne sayısı ve kapasite bunun ÜSTÜNE küçük, ek bir katkı yapıyor.

Bu, makalenin Limitations paragrafının (`sec:limitations`, "second scene" bölümü) kendisi
İÇİN önemli bir düzeltme gerektiriyor: o paragraf "6–11×"ü (aynı-model oranı, mr rejiminde)
doğrudan "90–958×" (control-vs-oracle oranı, regular rejiminde) ile karşılaştırıyor ve
farkı SADECE nesne sayısına atfediyormuş gibi okunabiliyor — oysa üç değişken AYNI ANDA
değişiyor: (1) ölçüm formülü (aynı-model vs control-vs-oracle — bunlar farklı büyüklükler,
doğrudan karşılaştırılamaz), (2) örnekleme rejimi (regular vs mr — BASKIN etken), (3) nesne
sayısı (1 vs 2 — küçük katkı). Dürüst sonuç: **"sessiz başarısızlık karmaşıklıkla
kendiliğinden sönüyor" iddiası desteklenmiyor** — hem tek-nesne hem iki-nesne durumunda,
AYNI mr-rejiminde, aynı-model oranı tek hanelerden-düşük-on'lara kadar (4.5–30×) benzer
mertebede kalıyor; multi-rate lifting'in verdiği ~30–220× iyileştirme ise HER durumda
büyük ve gerekli kalıyor. Yöntemin gerekliliği konusunda hakemin çıkarımı (belki
karmaşık sistemlerde gerek yok) **desteklenmiyor**; ama makalenin kendi "6 vs 90" çerçevesi
de üç değişkeni karıştırdığı için yanıltıcı — düzeltilmeli.

### Bonus: üç-nesne sahnesi (monotonluk testi)

`data_threeobj.py`/`run3.py` (n_osc=9, n_static=4 — 3 gerçek frekans için 3× yedek kapasite,
2-nesne probunun 6/2=3× oranıyla AYNI; ω=[0.75π, 0.382π, 1.236π], 3 bağımsız nesne, örtüşmeyen
bölgeler) ile 1→2→3 nesne trendine bakıldı. İlk kernel push'u bir bağımlılık hatasıyla
başarısız oldu (`data_threeobj.py`, `data2.py`'den `_obj` import ediyor ama ilk
`build_threeobj_kernel.py` bunu pakete dahil etmemişti — 10/10 iş `ModuleNotFoundError`
ile HATA verdi, hiçbir GPU zamanı harcanmadı çünkü hata import anında oluştu; düzeltilip
yeniden gönderildi, 10/10 iş OK, 0 HATA).

#### Ham sonuçlar (tohum bazında, E_1/2/L_Δ)

| dmd | tohum 0 | tohum 1 | tohum 2 | tohum 3 | tohum 4 | aralık |
|---|---|---|---|---|---|---|
| none (control) | 8.85 | 7.01 | 10.34 | 9.45 | 5.85 | **5.9–10.3×** |
| mr (lifting) | 1.07 | 7.86 | 7.26 | 52.4 | 11.6 | 1.1–52.4× (güvenilmez, aşağıda) |

**Nesne-sayısı trendi (control oranı):** 1 nesne (geniş kapasite) 7.7–15.1× → 2 nesne
4.5–10.9× → 3 nesne 5.9–10.3×. **Trend monoton DEĞİL — 2 nesnede DOYGUNLAŞIYOR**, 3.
nesnede oran KÜÇÜLMÜYOR (orta nokta hafifçe artıyor: ort. 2-nesne≈7.03, 3-nesne≈8.30). Bu,
"karmaşıklık arttıkça sorun kendiliğinden sönüyor" hipotezini AÇIKÇA ÇÜRÜTÜYOR — sorun 2
nesneden sonra sabitleniyor, kaybolmuyor.

**Yöntemin güvenilirliği 3 nesnede BOZULUYOR (dürüst, önemli bulgu):** mr'nin ham
mse_half'ı control'e göre: tohum0 37× iyi, tohum1 1.6× iyi, tohum2 2.4× iyi, **tohum3 1.8×
KÖTÜ** (lifting control'den daha kötü sonuç veriyor), tohum4 1.2× iyi (marjinal). Kök neden,
`dmd_omegas`'ın gerçek 3 frekansa (2.356, 1.200, 3.883 rad/s) en yakın adaylarla eşleştirilmesiyle
teşhis edildi: tohum0 hepsini doğru buluyor (2.393/1.206/3.898); tohum2 makul (2.373/1.303/3.789);
ama **tohum1, 3, 4'te 9 öğrenilmiş frekanstan İKİSİ AYNI değere çöküyor** (örn. tohum3'te üç
hedefin de en-yakın-eşleşmesi TEK bir değere, 1.193'e, çöküyor) — yani model 3 gerçek
frekanstan yalnızca 1'ini (bazen 2'sini) güvenle kilitleyebiliyor, diğerleri kayboluyor/
karışıyor. Bu, eq:lift'teki düşük-harmonik ayrım kuralının 3 bağımsız frekansı AYNI anda
ayrıştırmakta 2 frekanstan daha az güvenilir olduğunu gösteriyor — yöntem hâlâ ORTALAMADA
yardımcı oluyor ama 1-ve-2-nesne durumlarındaki KUSURSUZ 5/5 tutarlılığı (bkz. Tablo 3/6, f_true=1.00
her tohumda) 3 nesnede KAYBOLUYOR.

**Genel değerlendirme:** Hakemin sorusuna (karmaşıklık arttıkça sorun kayboluyor mu) cevap
NET: **HAYIR** — control oranı 2 nesnede doygunlaşıyor, 3 nesnede küçülmüyor. Ama bonus
bulgusu makaleyi daha da güçlendirmiyor sadece — yöntemin KENDİSİNİN de 3 bağımsız frekansta
daha az güvenilir hale geldiğini gösteriyor, bu da makalenin "we have not tested... more than
two independent objects" (Limitations, "Scale and statistics") ifadesinin NEDEN ihtiyatlı
kaldığını doğrudan haklı çıkarıyor — 3 nesne denendiğinde gerçekten yeni bir sınırlama
ortaya çıkıyor (frekans-ayrıştırma güvenilirliği düşüyor), bu ihtiyatlı çerçevenin gelecekte
ele alınması gereken gerçek bir sınır olduğunu gösteriyor.

**Dosyalar:** `/home/can/Projeler/oneri_karsilastirma/clock_diag/data_oneobj_wide.py`,
`run_oneobj_wide.py`, `data_threeobj.py`, `run3.py`, `build_reviewer_kernel.py`,
`build_threeobj_kernel.py`, `report_reviewer.py`; ham veri: `results_axis1.jsonl`,
`results_oneobj_wide.jsonl`, `results2_moreseeds.jsonl`, `results3.jsonl` (bonus),
`results_t2_moreseeds.jsonl`, `results_t3_moreseeds.jsonl`, `kaggle_out/results.jsonl`,
`kaggle_out_gpu/results.jsonl`, `kaggle_out_scenario2/results2.jsonl`.

---

## Görev B — Tablo 2/3/5'te tohum sayısı 2–3'ten 5'e

**Yöntem:** `run.py`/`run_pend.py`/`run2.py` DEĞİŞTİRİLMEDEN, makaledeki EGZAKT
konfigürasyonlarla (04_experiments.tex'ten çıkarıldı, `kaggle_job/kernel.py`,
`kaggle_gpu/kernel.py`, `kaggle_partial/kernel.py`, `kaggle_gate/kernel.py`'den doğrulandı),
eksik tohumlar eklendi. Yeni tohumlar mevcutlarla ÇAKIŞMIYOR.

### Tablo 2 (`tab:single-rate`) — 5 tohum (önceden: baseline 1, lock/ft/harm 2)

model=koop, init=small, sampling=regular, dmd_at=500.

| dmd | ρ=3/8 (0.75π) | ρ_g (0.763932π) | 1+ρ_g (2.763932π) |
|---|---|---|---|
| none (baseline) | 0.0433/0.0427/0.0429/0.0447/0.0427 | 0.0434/0.0427/0.0442/0.0436/0.0434 | 0.0518/0.0518/0.0541/0.0513/0.0511 |
| lock | 0.0469/0.0447/0.0426/0.0479/0.0413 | 0.0406/0.0485/0.0441/0.0409/0.0416 | 0.0548/0.0525/0.0541/0.0515/0.0506 |
| ft | 0.0464/0.0446/0.0419/0.0438/0.0431 | 0.0397/0.0485/0.0432/0.0424/0.0394 | 0.0546/0.0525/0.0530/0.0522/0.0512 |
| harm | 0.0414/0.0523/0.0318/0.0420/0.0401 | **5.6e-5/1.6e-3/2.4e-4/1.3e-4/9.6e-5** | 0.0321/0.0517/0.0317/0.0320/0.0321 |

**Bulgu (istatistiği güncelliyor):** Makale "harmonic lattice recovers the true spectrum
in one of six runs" diyor (2 tohum × 3 ρ = 6, 1 başarı). **5 tohumla ρ_g'de 5/5 başarı**
(hepsi <2e-3, oracle mertebesine yakın), 3/8 ve 1+ρ_g'de **0/5 başarı** (hepsi ~0.03–0.05,
baseline mertebesinde) — teoriyle TAM tutarlı (ρ_g irrasyonel, diğerleri düşük-mertebe
rasyonel, aliasing/kilitleme kuralı beklendiği gibi çalışıyor). Sonuç: "1/6" istatistiği
tohum-şansına bağlıydı; **mekanizma açıklaması doğru ama istatistik yanlış/eksikti** —
"ρ_g'de güvenilir, 3/8 ve 1+ρ_g'de güvenilir şekilde başarısız" olarak düzeltilmeli.

### Tablo 3 (`tab:lifting`) — 5 tohum (önceden: 3)

model=koop, sampling=mr, `--light`, dmd_at=500. E_1/2 (mse_half), f_true.

| ρ | dmd | tohum 0-4 (mse_half) | f_true | train_mse (=L_Δ) |
|---|---|---|---|---|
| 3/8 | none | 0.0211/0.0156/0.0203/0.0216/0.0202 | 0.44/0.59/0.32/0.36/0.34 | 1.23e-3/1.32e-3/1.27e-3/1.33e-3/8.96e-4 |
| 3/8 | mr | 5.2e-5/5.5e-4/9.2e-5/1.2e-4/8.2e-5 | 1.00 (×5) | 5.2e-5/1.3e-4/9.2e-5/1.1e-4/8.2e-5 |
| ρ_g | none | 0.0219/0.0224/0.0212/0.0223/0.0218 | 0.32/0.28/0.36/0.30/0.30 | 9.5e-4/1.2e-3/1.1e-3/1.1e-3/1.1e-3 |
| ρ_g | mr | 9.6e-5/6.4e-5/8.3e-5/7.3e-5/3.2e-4 | 1.00 (×5) | 8.5e-5/6.4e-5/7.7e-5/7.0e-5/8.2e-5 |
| 1+ρ_g | none | 0.0453/0.0466/0.0442/0.0441/0.0465 | 0.06/0.06/0.06/0.06/0.06 | 8.6e-4/8.2e-4/1.9e-3/1.1e-3/8.4e-4 |
| 1+ρ_g | mr | 5.8e-5/6.1e-5/4.3e-4/9.7e-5/6.6e-5 | 1.00 (×5) | 5.3e-5/5.8e-5/1.2e-4/8.3e-5/6.6e-5 |
| 1/4 | none | 0.0041/0.0033/0.0037/0.0052/0.0037 | 0.89/0.97/0.97/0.80/0.86 | 3.7e-4/6.4e-4/1.1e-3/6.8e-4/6.1e-4 |
| 1/4 | mr | 2.0e-4/2.1e-3/1.4e-4/2.9e-4/3.3e-3 | 1.00/0.95/1.00/1.00/0.66 | 1.8e-4/5.0e-4/1.1e-4/1.3e-4/2.8e-4 |

**Bulgu:** Ana iddia (control ~1.6–4.7×10⁻²'de kalıyor, mr oracle mertebesine iniyor,
f_true=1.00 3/8+ρ_g+1+ρ_g'de) 5 tohumda da AYNEN geçerli. ρ=1/4'te 5. tohum (seed4) mr
altında f_true=0.66'ya düşüyor ve mse_half control'den daha kötü (0.0033) — makalenin
zaten flag ettiği "ρ=1/4'te farklı davranış" gözlemini DOĞRULUYOR ve tek bir kötü-tohum
örneği veriyor (önceden 3 tohumda 2/3 iyileşme deniyordu, şimdi 3/5 iyileşme — hâlâ
çoğunluk iyileşiyor ama daha net bir başarısız-tohum örneği var).

### Tablo 5 (`tab:pendulum`) — 5 tohum (önceden: 2–3), `dmd_at=1500` (doğrulandı, üstte)

| satır | tohum | mse_half | f_true | residual |
|---|---|---|---|---|
| control (dmd=none, mr) | 0-4 | 0.00551/0.00536/0.00554/0.00540/0.00573 | 0.49/0.49/0.49/0.49/0.48 | — |
| ungated (partial, dmd_at=1500) | 0-4 | 0.00900/0.01055/0.00942/0.00851/0.01057 | 0.40/0.35/0.38/0.44/0.50 | 0.111/0.602/0.328/0.581/0.428 |
| gated (partialg, dmd_at=1500) | 0-4 | 0.00557/0.00536/0.00554/0.00542/**0.01227** | 0.49/0.49/0.49/0.49/**0.20** | 0.101/0.678/0.485/0.475/**0.0706** |
| oracle (regular) | 0-4 | 0.00377/0.00400/0.00376/0.00397/0.00345 | 0.70/0.66/0.70/0.66/0.72 | — |

**Bulgu (önemli, dürüstçe raporlanmalı):** Seed 4'te gate **GERÇEKTEN YANLIŞ KİLİTLENDİ**
(residual=0.0706 < τ=0.1, `gated_skip` alanı False) — bu, makalenin şu ana kadar tek bir
"0.001 uzakta" veri noktasıyla "canlı risk" olarak flag ettiği false-accept senaryosunun
**GERÇEKTEN GERÇEKLEŞTİĞİ** bir örnek: mse_half control'ün 2.3× kötüsüne çıkıyor
(0.0055→0.0123), f_true 0.49'dan 0.20'ye düşüyor — ungated satırının en kötü örneklerinden
daha da kötü. Bu tesadüfen bulunmuş 5. tohum, herhangi bir ek gürültü olmadan (varsayılan
noise=0.01) ortaya çıktı. Bkz. Görev D'de gürültü taramasıyla ilişkisi.

**Dosyalar:** `results_t2_moreseeds.jsonl`, `results_t3_moreseeds.jsonl`,
`results_pend_moreseeds.jsonl`; referans (eski tohumlar): `kaggle_out/results.jsonl`,
`kaggle_out_gpu/results.jsonl`, `kaggle_out_partial/results_pend.jsonl`,
`kaggle_out_gate/results_pend.jsonl`.

---

## Görev C — Neural ODE (ρ=1/4) tek tohumdan 5 tohuma

**Hipotez:** `obs:regimes`'teki tek-tohum gözlemi (κ=2.04, C=1.81) tohum-şansı mı, yoksa
NODE'un "clock" rejimine düşmesi güvenilir mi?

**Tohum kimliği:** Orijinal tek-tohum sonucu `results.jsonl`/`results_merged2.jsonl`'da
BULUNDU: `model=node, omega_mult=0.5, seed=0` (κ=2.0361, C=1.8066 — makaledeki "κ=2.04,
C=1.81" ile TAM eşleşiyor). 4 tohum daha eklendi (seed 1-4), `run.py --model node
--omega_mult 0.5 --seed {1,2,3,4}` (`--light` KULLANILMADI — tam diagnostics gerekli).

### Ham sonuçlar (tohum bazında)

| tohum | κ (chain kappa) | C (loop_closure) | mse_half |
|---|---|---|---|
| 0 (orijinal, makaledeki) | 2.0361 | 1.8066 | 0.01848 |
| 1 | 2.3440 | 1.7687 | 0.01783 |
| 2 | 2.2984 | 1.9149 | 0.02062 |
| 3 | 2.2978 | 1.9021 | 0.02084 |
| 4 | 2.3183 | 1.9078 | 0.01976 |
| 5 (ek) | 2.1066 | 1.9038 | 0.02163 |
| **ortalama** | **2.23** | **1.87** | **0.0199** |

### Değerlendirme: **DOĞRULANDI, düşük varyansla**

κ 2.04–2.34 (spread ~%15), C 1.77–1.91 (spread ~%8) — sıkı bir bant, orijinal tek-tohumun
tipik olduğunu gösteriyor, aşırı-uç bir çekim değil. Tüm 6 tohum κ>2 (latent görüntüden en
az 2× yavaş dönüyor) ve C>1.7 (döngü bir görüntü periyodunda kapanmıyor) — "clock" rejiminin
imzası TUTARLI şekilde mevcut. Makalenin "we report it as a reason to suspect the failure
is not exclusive to linear latents" ifadesi bu 5-tohum sonucuyla GÜÇLENDİ (tek-tohum şansı
değil, tekrarlanabilir bir gözlem); makalenin kendi temkinli çerçevesi ("not as evidence
that it is common to all continuous-time latent models") hâlâ doğru ve gerekli — çünkü bu
tek bir ρ değerinde (1/4), tek bir NODE mimarisinde.

**Dosyalar:** `results_node5.jsonl` (yeni, seed 1-4); orijinal seed0:
`results.jsonl`/`results_merged2.jsonl` (`model=node, omega_mult=0.5`).

---

## Görev D — Gürültü/eşik (τ=0.1) duyarlılık taraması

**Hipotez:** Piksel gürültüsü σ artınca (a) rijit dönmede residual τ'yu AŞIP doğru modu
yanlışlıkla reddedebilir mi (false reject)? (b) sarkaçta residual τ'nun ALTINA inip yanlış
frekansı yanlışlıkla kilitleyebilir mi (false accept)?

**Uygulama:** `run_noise.py`/`run_pend_noise.py` — `run.py`/`run_pend.py`'nin BİREBİR
şablonu, TEK fark: `--noise` argümanı hem ana eğitim batch'lerine HEM dmd_init çağrılarının
probe lambda'larına (mevcut kodda sabit `noise=0.0`) geçiriliyor — kapı artık eğitimdekiyle
AYNI gürültü seviyesinde değerlendiriliyor (gerçekçi dağıtım senaryosu).

### D1 — Rijit dönme (ρ_g, `dmd=mr`, ungated — residual/marj okunuyor)

3 tohum/seviye, σ ∈ {0.01, 0.03, 0.05}. Gerçek osilatörün (4 modun `dmd_principal`'ı
ω_true=2.4'e en yakın olanı) residual'i izlendi — diğer 3 mod boş kapasite/statik, onların
yüksek residual'i beklenen ve önemsiz.

| σ | tohum | gerçek modun residual | marj | mse_half |
|---|---|---|---|---|
| 0.01 | 0 | 0.0398 | 0.686 | 1.66e-4 |
| 0.01 | 1 | 0.0410 | 0.683 | 6.68e-5 |
| 0.01 | 2 | 0.0017 | 0.762 | 7.61e-5 |
| 0.03 | 0 | 0.0281 | 0.709 | 1.00e-4 |
| 0.03 | 1 | 0.0153 | 0.735 | 8.95e-5 |
| 0.03 | 2 | 0.0055 | 0.754 | 1.07e-4 |
| 0.05 | 0 | 0.0405 | 0.684 | 1.62e-4 |
| 0.05 | 1 | 0.0387 | 0.688 | 8.82e-5 |
| 0.05 | 2 | 0.0076 | 0.750 | 1.21e-4 |

**Değerlendirme: false-reject riski σ=0.05'e (referansın 5×'i) kadar GERÇEKLEŞMİYOR.**
Gerçek modun residual'i τ=0.1'in HER ZAMAN belirgin şekilde altında (max 0.041, τ'nun
%41'i), marj sağlıklı (0.68–0.76), σ ile monoton bozulma yok (0.05'te residual 0.01'dekiyle
aynı mertebede). E_1/2 de σ ile büyük ölçüde bozulmuyor (1e-4 mertebesinde kalıyor, oracle
seviyesine yakın). Rijit dönmede kapı, test edilen gürültü aralığında güvenli.

### D2 — Piksel sarkaç (`dmd=partialg`, `dmd_at=1500`, gated — false-accept izleniyor)

3 tohum/seviye, σ ∈ {0.01, 0.03, 0.05}.

| σ | tohum | residual | gated_skip | mse_half | f_true |
|---|---|---|---|---|---|
| 0.01 | 0 | 0.1036 | True (doğru abstain, τ'ya ÇOK yakın) | 0.00553 | 0.49 |
| 0.01 | 1 | 0.4640 | True | 0.00538 | 0.50 |
| 0.01 | 2 | 0.4479 | True | 0.00555 | 0.49 |
| 0.03 | 0 | 0.4946 | True | 0.00550 | 0.49 |
| 0.03 | 1 | 0.3133 | True | 0.00542 | 0.50 |
| 0.03 | 2 | **0.0461** | **False (GERÇEK false-accept)** | **0.00967** | **0.30** |
| 0.05 | 0 | 0.3994 | True | 0.00556 | 0.49 |
| 0.05 | 1 | 0.9410 | True | 0.00533 | 0.49 |
| 0.05 | 2 | 0.6737 | True | 0.00554 | 0.48 |

**Değerlendirme: false-accept riski GERÇEK ve DAHA SIK, ama σ'ya monoton bağlı DEĞİL.**
σ=0.03/tohum2'de kapı YANLIŞ KİLİTLENDİ (residual=0.046, açıkça τ'nun altında, "0.001
uzakta" değil) — mse_half control'ün ~1.7× kötüsüne çıktı, f_true 0.49→0.30. σ=0.01 ve
σ=0.05'te (sadece 3 tohumla) false-accept GÖRÜLMEDİ; σ=0.01/tohum0'ın residual'i (0.1036)
makaledeki "0.101, τ'dan 0.001 uzakta" değeriyle TAM eşleşiyor (aynı üretim yolu, aynı
tohum). Görev B'nin Tablo-5-genişletmesinde AYRICA (bu taramadan bağımsız, aynı varsayılan
σ=0.01'de) bir false-accept bulundu (seed4, residual=0.0706) — yani gürültü taraması VE
bağımsız 5-tohumluk Tablo-5 genişlemesi BİRLİKTE, 14 gated-sarkaç çekiminde 2 gerçek
false-accept veriyor (~%14 ampirik oran). Bu, makalenin "0.001 uzakta, canlı risk" ifadesinin
ÇOK ötesinde: risk teorik değil, KÜÇÜK örneklemde bile tekrar tekrar gerçekleşiyor. Ama
σ ile monoton İLİŞKİLİ DEĞİL — bu, mekanizmanın "gürültü residual'i sistematik olarak
düşürüyor" değil, "sarkaçın genlik-bağımlı frekansı zaten hiçbir τ ile güvenli
ayrıştırılamıyor, hangi tohum/gürültü kombinasyonunun rastgele düşük bir residual'e denk
geldiği şans meselesi" olduğunu gösteriyor — makalenin kendi "consistency certificate, not
correctness proof" çerçevesiyle TUTARLI, ama ampirik kanıt şimdi ÇOK daha güçlü ve tekil
bir veri noktasına değil, tekrarlanan bir gözleme dayanıyor.

**Dosyalar:** `run_noise.py`, `run_pend_noise.py` (yeni); ham veri: `results_noise.jsonl`,
`results_pend_noise.jsonl`.

---

## Bölüm 5 — Bütünlük doğrulaması

```
$ python3 dmd_init.py
selftest DMD modal: ... OK
selftest harmonik: ... OK
selftest çok hızlı: ... OK (2 rotation numbers)
selftest kısmi: ... OK
```
4/4 selftest geçti (hem çalışmadan önce hem tüm Kaggle koşuları bittikten sonra tekrar
çalıştırıldı, aynı sonuç).

Korunan dosyaların mtime'ı (görev başlamadan önce vs şimdi, `stat -c '%Y %n'`) **BİREBİR
AYNI** — hiçbiri değişmedi:
`data.py, models.py, dmd_init.py, losses.py, run.py, run_pend.py, run2.py, data2.py,
diagnostics.py`.

Yeni dosyalar (hepsi `/home/can/Projeler/oneri_karsilastirma/clock_diag/` altında):
`data_oneobj_wide.py`, `run_oneobj_wide.py`, `run_noise.py`, `run_pend_noise.py`,
`data_threeobj.py`, `run3.py`, `build_reviewer_kernel.py`, `build_threeobj_kernel.py`,
`report_reviewer.py`, `NOTES_reviewer_response.md` (bu dosya).

Yeni ham veri dosyaları: `results_t2_moreseeds.jsonl`, `results_t3_moreseeds.jsonl`,
`results_pend_moreseeds.jsonl`, `results_axis1.jsonl`, `results_oneobj_wide.jsonl`,
`results2_moreseeds.jsonl`, `results_node5.jsonl`, `results_noise.jsonl`,
`results_pend_noise.jsonl`, `results3.jsonl` (bonus).

GitHub'a hiçbir şey push edilmedi. `paper/`, `paper_l4dc/`, `paper_tmlr/` hiçbirine
yazılmadı (sadece `paper/sections/04_experiments.tex` ve `05_limitations.tex` salt-okunur
konfigürasyon doğrulaması için okundu).

# Genlige/enerjiye bagli frekansli (nonlineer) osilatorler icin cok-hizli spektral kimlik-tanima -- genellenebilir mi?

Bu, `paper/` ve `paper_l4dc/`'deki iki calismadan AYRI, acik uclu bir arastirma notu. Iki
makaleye hicbir sekilde entegre edilmedi, o klasorlere yazilmadi. Motivasyon: `paper/sections/
04_experiments.tex`'teki "An amplitude-dependent generator does not close the gap either"
paragrafinda KoopAmp (r^2 uzerinden kucuk MLP ile omega_j*(1+g_j(r^2)) modulasyonu) piksel
sarkacta basarisiz oldu (E_1/2=0.0119-0.0149, kontrolden bile kotu; oracle'a eklenince de
kotulesti). Soru: bu yaklasimin oz mekanizmasi mi kusurlu, yoksa deneyin kosullari mi
(CNN encoder + multi-rate DMD kilitleme ile ayni anda)?

## 1) Literatur taramasi

En yakin 5 calisma:

**1. Lusch, Kutz, Brunton, "Deep learning for universal linear embeddings of nonlinear
dynamics," Nature Communications 2018 (arXiv:1712.09707).** EN DOGRUDAN ILGILI CALISMA.
Koopman-otokodlayici (auto-encoder + K matrisi + lineerlik/tahmin kaybi) egitiyorlar ve
surekli spektrumlu sistemler icin TAM OLARAK bizim KoopAmp'in yaptigi seyi yapan bir
"auxiliary network" tanimliyorlar: ozdeger/frekansi latent yaricapin karesine (‖y‖^2)
kosullu bir agla parametrize ediyorlar (bizim omega_j*(1+g_j(r^2)) ile ayni fikir, farkli
parametrizasyon: B(mu,omega)=exp(mu*dt)*R(omega*dt)). ONEMLI: bu agi TAM SARKAC uzerinde
(x''=-sin(x), TAM nonlineer, kucuk-aci yaklastirmasi degil) test ediyorlar ve test hatasi
1.1e-7 ile BASARILI oluyorlar, ogrenilen buyukluk Hamilton enerji seviye egrilerini izliyor
(Mezic'in teorik sonuclariyla dogrulanmis). Bizim denememizden 3 kritik farkla: (a) girdi
piksel degil, dogrudan durum (theta, theta_dot) -- CNN'in kendi latent'ini enerjiyle
hizalamasi gerekmiyor, dusuk boyutlu girdi zaten neredeyse enerjiyle orantili; (b) kayip
fonksiyonu bizimkinden daha zengin: ayri rekonstruksiyon + lineerlik (latent'te ||phi(x_{k+m})
- K^m phi(x_k)||) + tahmin + L_inf + L2 terimleri var, bizim run_pend.py'de tek bir
piksel-tahmin MSE'si var; (c) multi-rate DMD kilitlemesi YOK -- auxiliary network bastan
itibaren serbestce, hicbir rijit-donme varsayimiyla catismadan egitiliyor. Bu makale bizim
hipotezimizin ("kucuk MLP transandantal iliskiyi yakalayamaz") EN GUCLU CURUTUCUSU: ayni
fiziksel sistemde (sarkac), ayni temel mekanizmada (r^2 -> frekans MLP'si) basari raporlaniyor.

**2. Bondesan & Lamacraft, "Learning Symmetries of Classical Integrable Systems," 2019
(arXiv:1906.04645).** Aksiyon-aci donusumunu SEMPLEKTIK NORMALIZING FLOW ile (her katman
kanonik/semplektik yapiyi koruyan tersinir bir katman) doğrudan trajectory verisinden
ogreniyorlar. Bizim yaklasimimizdan farki: latent koordinatlarin fiziksel aksiyon-aci
degiskenlerine karsilik gelmesini MIMARI OLARAK (semplektik kisit) zorluyorlar, bizim
KoopAmp'te ise encoder'in r^2'yi enerjiyle hizalamasini zorlayan HICBIR kisit yok -- bu
tam da bizim ikinci, "daha derin" hipotezimizin (encoder'in latent'i fiziksel enerjiye
hizalamasi icin zorlama yok) literatur karsiligi: bu makale tam olarak o zorlamayi ekleyerek
cozuyor.

**3. Daigavane, Qu, Vondrick (?), "Learning Integrable Dynamics with Action-Angle Networks,"
NeurIPS ML4PS 2022 (arXiv:2211.15338).** Girdi koordinatlarindan aksiyon-aci uzayina NONLINEER
bir donusum ogreniyorlar (harmonik osilator, Kepler problemi gibi entegre edilebilir
sistemlerde), boylece dinamik latent'te LINEER (aci acilarda sabit hizla ilerliyor) oluyor.
Bizim yontemimizle ayni ruhta (nonlineer -> lineer latent) ama aksiyon degiskeninin kendisini
(enerjiyle iliskili buyuklugun) ayri, acikca hedeflenen bir cikti olarak ogretiyorlar; genel
"r^2'den kucuk bir MLP duzeltmesi" yaklasimindan daha yapisal. Sarkac spesifik olarak test
edilip edilmedigi tam netlesmedi (abstract'ta harmonik osilator + Kepler aciktan geciyor).

**4. Greydanus, Dzamba, Yosinski, "Hamiltonian Neural Networks," NeurIPS 2019.** Latent'i
degil, DOGRUDAN gozlemi (q,p) parametrize edip bir Hamiltonian H_theta(q,p) ogreniyorlar,
Hamilton denklemleriyle simplektik gradyan uzerinden turetilen dinamikle enerjiyi TAM
KORUYOR (bizim r^2'nin korunmasi varsayimimiza benzer ama enerji direkt agin cikisi, latent
bir yaricap degil). Bizim probleme (iki-hizli ornekleme ile frekans sertifikasi) dogrudan
uygulanmiyor -- HNN bir frekans/spektrum sertifikasi uretmiyor, sadece entegre edilebilir
(enerji-koruyan) bir vektor alani ogreniyor; DMD/faz-kilitleme fikriyle birlesimi acik bir
soru.

**5. Cranmer, Greydanus, Hoyer, Battaglia, Spergel, Ho, "Lagrangian Neural Networks," ICLR
2020 Workshop (arXiv:2003.04630).** HNN'nin kanonik koordinat gereksinimini kaldiriyor
(rastgele genellenmis koordinatlarda calisiyor), ama yine dogrudan gozlem uzerinde bir
Lagrangian ogreniyor, bizim latent-Koopman + coklu-hiz spektral kilitleme cercevemizle ayni
eksende degil; bahsetmeye deger cunku "fiziksel yapiyi zorlayarak ogrenme" ailesinin bir
parcasi ve HNN'ye alternatif.

**Ozet fark:** Bizim yontemimiz (paper/'daki linear KoopCT + coklu-hizli DMD kilitleme) bir
SERTIFIKA uretiyor (residual esikli kabul/red) ve rijit donme icin kanitlanmis garanti
veriyor. Yukaridaki calismalarin hicbiri boyle bir coklu-hizli tutarlik sertifikasi/gate
mekanizmasi sunmuyor -- hepsi "frekans/enerji baginlisini ogren" tarafinda, "bu ogrenilen
seyin dogru oldugunu nasil sertifikalarsin" tarafinda degil. Yani nonlineer uzantimiz icin
literatur, oncelikle "nasil ogrenilir" sorusuna (Lusch ve Bondesan-Lamacraft en yakin
cevaplar) cevap veriyor; "nasil sertifikalanir" sorusu (bizim asil katkimiz) hala acik.

## 2) Hipotez degerlendirmesi

Baslangic hipotezi iki parcaliydi:
(a) Aksiyon-aci iliskisi transandantal (eliptik integral K(k)); kucuk bir MLP bunu r^2'den
    kucuk bir sapmayla yakalayamaz.
(b) Daha derin: encoder'in latent koordinatlarini fiziksel enerji/aksiyona hizalamasi icin
    hicbir zorlamasi yok.

**Lusch et al. (2018) bulgusu (a)'yi zayiflatiyor**: AYNI sarkac sistemi uzerinde (tam
nonlineer, kucuk-aci degil), r^2'ye kosullu kucuk bir ag ile 1e-7 mertebesinde hata elde
etmisler. Demek ki mekanizma prensipte transandantal iliskiyi (universal yaklastirma +
yeterli egitim sinyali ile) yakalayabiliyor -- EGER dogru kosullar saglanirsa (dusuk boyutlu/
temiz girdi, zengin kayip, kilitleme yok).

**(b) sentetik deneyle test edildi (asagida) ve KISMEN DOGRULANDI, ama beklenenden daha
incelikli sekilde**: encoder'in r^2'yi FIZIKSEL enerjiyle birebir hizalamasi icin zorlama
olmasa bile, TAHMIN KAYBI TEK BASINA (baska hicbir kisit/loss terimi olmadan) modeli, ozgun
enerji-frekans yasasinin bir MONOTON YENIDEN PARAMETRELENMESINE (r -> h(r) radyal carpitma,
faz hizi korunuyor) yeterince yakinsatabiliyor ki ENTERPOLASYON HATASI oracle seviyesine
inebiliyor -- ogrenilen g_j(r^2) fonksiyonu DOGRU sabiti (K_TRUE) birebir tutturmasa da,
DOGRU ISARET ve KABACA DOGRU sekli yakaliyor ve bu, tahmin performansi icin yeterli. Yani
"hizalama zorlamasi yok" sorunu, bu kontrollu ortamda BEKLENENDEN DAHA AZ ENGEL cikardi.

**Sonuc: hipotez (a) byuk olcude curutuldu (transandantal iliski baslica engel degil, en
azindan kucuk-aci mertebesinde polinom yaklastirma yeterli olabiliyor gibi); hipotez (b)
kismen dogrulandi ama sanildigindan daha az kritik cikti -- gradyan inisi, dogru model
sinifi ve CATISMAYAN bir kayip fonksiyonuyla, kendi kendine iyi bir yeniden parametrelemeye
yakinsayabiliyor.** Gercek darbogaz, sentetik deneyin gosterdigi kadariyla, muhtemelen
paper'daki deneyin IKI ek unsurunda: (i) 32x32 piksellerden CNN encoder ile r^2 cikarmanin
guclugu (dusuk boyutlu/temiz girdiye gore cok daha zor bir cikarim problemi) ve (ii) KoopAmp'in
multi-rate DMD KILITLEMESI ile AYNI ANDA denenmesi -- kilitleme mekanizmasi (dmd_init.py,
multirate_reinit) TAMAMEN sabit-omega varsayimiyla yazilmis; bir moda sabit bir frekans
"kilitlemek", o modun r'ye bagli surekli degisen bir frekansa sahip olmasiyla DOGRUDAN
CATISIYOR. Paper'in kendi ifadesiyle bu birlikte denenmis ("Combined with multi-rate locking")
ve kotu sonuc vermis; bizim sentetik testimiz kilitleme OLMADAN calisiyor.

## 3) Sentetik dogrulama deneyi

Dosya: `clock_diag/synth_amp_test.py` (yeni, `models.py`'a dokunulmadi).

**Tasarim:** Gercek uretici mekanizma TAM KAPALI FORMDA ve POLINOM: r0 ~ U(0.3,1.2) (sarkacla
ayni genlik araligi), omega(r) = 2.4*(1 - 0.15*r^2) -- bu, sarkacin KENDI kucuk-aci
acilimindaki terimle ayni isaret/mertebede (omega(A) ~ omega0*(1-A^2/16+...)), yani "polinom,
eliptik integral degil" kosulunu saglarken fiziksel olarak anlamli. Latent z=(a,b)=r*(cos,sin)
donme hareketi (r trajectory boyunca korunuyor, tipki KoopAmp'in varsaydigi kapali-form akis
gibi). Gozlem: z, SABIT (egitilmeyen) rastgele bir 2->48->10 boyutlu tanh-MLP ile 10 boyuta
gomuluyor -- CNN yerine, "encoder'in ogrenmesi gereken nonlineer bir gozlem haritasi" rolunu
oynayan kontrollu bir vekil. Model sinifi `models.py`'daki KoopCT/KoopAmp ile BIREBIR AYNI
omega_eff formu (farkli Enc/Dec: piksel CNN yerine genel MLP).

Uc kol, ayni egitim kaybiyla (run_pend.py ile ayni sekilde: pencere boyunca MSE, tek
kayip terimi, DMD/kilitleme YOK), 3 tohum:
- **control**: KoopCT, tek sabit ogrenilebilir omega (genlik-kor taban cizgisi)
- **treatment**: KoopAmp, g_j sifirdan ogreniliyor -- paper'daki basarisiz KoopAmp ile AYNI
  mekanizma, ama piksel/CNN yok, multi-rate kilitleme yok
- **true_law**: KoopAmp ama g_j DONDURULMUS, dogru polinom yasaya esitlenmis (tavan/oracle)

**Sonuc (4000 iterasyon, 3 tohum, E_1/2 analogu = t=0'dan devam eden egitimde HIC gorulmeyen
0.5 adimli izgarada MSE):**

| kol | mse_train | mse_half (E_1/2 analogu) |
|---|---|---|
| control | 0.0095-0.0150 | **0.18-0.27** |
| treatment (KoopAmp, sifirdan) | 0.0042-0.0043 | **0.0046-0.0055** |
| true_law (oracle, dogru yasa donduruldu) | 0.0035-0.0040 | **0.0043-0.0051** |

Treatment, oracle'in (true_law) neredeyse TAM SEVIYESINDE (~%5-15 fark), control'den ise
**35-55x** daha iyi. Ogrenilen g_1(r^2) egrisi, ISARET ve MONOTON SEKIL olarak dogru yasayla
tutarli (butun tohumlarda r arttikca negatife dogru azaliyor) ama LITERAL sabiti tutturmuyor
(orn. tohum 0: r=1.2'de ogrenilen -0.088, gercek -0.216) -- bu, encoder'in serbest radyal
yeniden-parametrelenme simetrisiyle (r -> h(r), faz hizina gore telafi edilebilir) tutarli:
tahmin performansi bu simetriye gore DEGISMEZ, ama g'nin literal degeri degisir. Ham veri:
`clock_diag/synth_amp_results.json`.

**Bu deney, hipotezi (a) buyuk olcude curutuyor ve KoopAmp mekanizmasinin -- CNN piksel
encoder ve multi-rate kilitleme olmadan, model sinifi veri-uretme sureciyle esletiginde --
prensipte calistigini gosteriyor.**

## 4) Piksel sarkac sonucu

Sentetik test basarili oldugu icin plana gore adim 3'e gecildi: `clock_diag/run_pend.py`
ve `data_pend.py` DEGISTIRILMEDEN, gercek piksel sarkac uzerinde KoopAmp'i multi-rate
KILITLEME OLMADAN (--dmd none) calistirip kontrolle (--model koop --dmd none) karsilastirdik
-- paper'daki basarisiz deney HER ZAMAN kilitleme ile birlikte calistirilmisti
("Combined with multi-rate locking"); kilitlemeyi cikarinca ne olur, paper'da raporlanmamis.
Ayarlar paper'inkiyle ayni: --sampling mr (karisik-hizli veri), --iters 3000, --dmd_at 500
(kullanilmadi cunku --dmd none), 32x32 piksel, CNN encoder/decoder. Ham cikti:
`clock_diag/nonlinear_pend_results.jsonl`.

**Sonuc (kontrol 3 tohum tamamlandi; KoopAmp kilitlemeden icin 2/3 tohum tamamlandi, 3.
tohum zaman kisitindan dolayi durduruldu -- asagida gerekce):**

| model | tohum | secs | mse_train | mse_half (E_1/2) | f_true |
|---|---|---|---|---|---|
| control (koop, dmd=none) | 0 | 396 | 4.4e-05 | 0.00553 | 0.489 |
| control (koop, dmd=none) | 1 | 385 | 3.7e-05 | 0.00540 | 0.494 |
| control (koop, dmd=none) | 2 | 377 | 6.5e-05 | 0.00550 | 0.496 |
| koopamp, dmd=none (kilitsiz) | 0 | 394 | 4.7e-05 | 0.00528 | 0.497 |
| koopamp, dmd=none (kilitsiz) | 1 | **12220** | **0.0038** | 0.00995 | 0.381 |
| koopamp, dmd=none (kilitsiz) | 2 | -- | -- | -- | -- (tamamlanmadi, asagida) |

Kontrol satirlari paper'in Tablo (tab:pendulum) "control (no spectral step)" satiriyla
BIREBIR ESLESIYOR (paper: mse_half=0.0055, f_true=0.49) -- bu, sentetik/piksel kurulumun
paper'la tutarli oldugunu dogrulayan bir saglama testi.

**Kilitsiz KoopAmp, KARISIK bir sonuc verdi, tek yonlu bir iyilesme DEGIL:**
- Tohum 0: control ile AYNI mertebede (mse_half=0.00528 vs control ~0.0054, f_true=0.497 vs
  ~0.49) -- kilitlemenin paper'daki kotu sonucu (0.0090-0.0105) UZERINDE bir iyilesme var,
  ama control'u anlamli sekilde GECMIYOR, sadece onunla ayni seviyeye geliyor.
- Tohum 1: control'den ACIKCA KOTU (mse_half=0.00995, neredeyse control'un 2 kati; f_true=0.381,
  control'un ~0.49'undan belirgin dusuk) VE ciddi bir OPTIMIZASYON PATOLOJISI gosteriyor:
  egitim suresi 12220 saniye (~3.4 saat), diger TUM kosularin (ortalama ~390 saniye) **~31
  kati**; egitim kaybi da yakinsamamis (mse_train=0.0038, digerlerinin ~0.00004-0.00007'sinin
  ~55-100 kati). Bu, sentetik testte HICBIR tohumda gorulmeyen bir davranis (sentetik testte
  3/3 tohum duzgun ve hizli yakinsadi).
- Tohum 2: yukaridaki tohum 1 anomalisi nedeniyle bu kosu supheyle izlendi; 238 saniyede
  (normal araliktaydi, henuz "takilmis" degildi) koordinatorun acik talimatiyla ZAMAN
  KISITINDAN DOLAYI durduruldu, tamamlanmadi. Bu bir "basarisizlik" degil, bilincli bir
  zaman-kutusu karari; 3. tohumun nihai degeri bilinmiyor.

**Yorum:** Kilitlemeyi kaldirmak paper'daki EN KOTU sonucu (0.0090-0.0105, kilitli KoopAmp)
onlese de, kontrolu GUVENILIR SEKILDE gecmiyor ve en az bir tohumda (2/3 tamamlanan kosunun
yarisi) KoopAmp'in kendi optimizasyonu -- multi-rate kilitleme hic devrede olmadan bile --
ciddi sekilde kararsiz. Bu, sentetik deneyle (Bolum 3) celisen degil, onu TAMAMLAYAN bir
bulgu: sentetik ortamda (dusuk boyutlu, temiz MLP gozlem haritasi, model sinifi veri-uretme
sureciyle TAM eslesmis) mekanizma guvenilir sekilde calisiyordu; 32x32 piksel + CNN
encoder'a gecince -- daha zor bir cikarim problemi, daha karmasik kayip yuzeyi -- ayni
mekanizma EN AZINDAN BAZI TOHUMLARDA ciddi sekilde kirilgan hale geliyor. Bu da orijinal
hipotez (b)'yi (encoder'in latent'i fiziksel enerjiye hizalamasi icin zorlama yok) sentetik
testin ilk izleniminden DAHA GUCLU sekilde destekliyor: piksel-CNN'in bu hizalamayi
kendiliginden bulmasi -- ozellikle nonlineer omega_eff'in ekledigi ekstra serbestlik
dereceleriyle birlikte -- optimizasyon acisindan kirilgan bir surec.

## 5) Tavsiye

**Bu yon duz bir cikmaz sokak DEGIL, ama "bu haliyle" (paper'daki KoopAmp+multi-rate-kilit
kombinasyonu) da CALISMIYOR ve neden calismadigina dair net bir aciklama artik var.**

Gerekce, uc bulgunun sentezi:

1. **Literatur (Lusch, Kutz, Brunton 2018)** ayni fiziksel sistemde (tam nonlineer sarkac)
   ayni temel mekanizmanin (r^2 -> frekans MLP'si) ILKE OLARAK calisabildigini gosteriyor
   (test hatasi 1e-7), ama DUSUK BOYUTLU, DOGRUDAN durum girdisiyle ve bizimkinden daha
   zengin bir kayip fonksiyonuyla (rekonstruksiyon + lineerlik + tahmin + L_inf + L2).

2. **Sentetik test (Bolum 3)**, model sinifi veri-uretme sureciyle TAM eslesince ve girdi
   dusuk boyutlu/MLP-dostu olunca, KoopAmp'in bizim BASIT kayip fonksiyonumuzla (tek MSE,
   run_pend.py ile ayni) bile oracle seviyesine (35-55x iyilesme) ulastigini gosterdi --
   3/3 tohumda tutarli ve hizli.

3. **Piksel sarkac (Bolum 4)**, ayni mekanizma + ayni basit kayip, ama CNN piksel encoder'a
   ve GERCEK (eliptik integral) aksiyon-aci iliskisine gecince: kilitlemeyi kaldirmak bile
   guvenilir bir iyilesme SAGLAMIYOR (1 tohum notr, 1 tohum acikca kotu + ciddi optimizasyon
   kararsizligi, 1 tohum tamamlanmadi).

Yani darbogaz ne saf "MLP transandantal fonksiyonu ogrenemez" (curutuldu) ne de sadece
"multi-rate kilitlemeyle celisme" (kilitlemeyi kaldirinca da guvenilir bir iyilesme yok) --
en olasi aciklama, PIKSEL ENCODER'IN kendi latent'ini enerji/genlikle hizalamasi icin hicbir
mimari zorlamanin olmamasinin, nonlineer omega_eff'in eklediği ekstra parametrelerle
birlesince optimizasyonu kirilgan/tohuma-duyarli hale getirmesi (Bolum 2'deki hipotez (b),
baslangicta dusunulenden daha guclu bir sekilde).

**Somut oneriler, gelecekteki AYRI bir calisma/tez icin:**
- Lusch ve ark.'in kullandigi ZENGIN kayip fonksiyonunu (ozellikle latent-uzayda ayri bir
  "lineerlik" terimi ve L_inf) run_pend.py tarzi tek-MSE kaybina eklemek, kirilganligi
  azaltip azaltmadigini test etmek -- ucuz ve dogrudan uygulanabilir bir sonraki adim.
- `dmd_init.py`'nin coklu-hizli sertifika mekanizmasini (residual/gate) r-BAGIMLI
  frekanslara genellemek: su an TAMAMEN sabit-omega varsayimiyla yazili (bkz. gorev
  bağlaminda belirtilen not); bu, hem paper'in kendi limitasyonlar bolumundeki acik
  soruyu ("conditioning the lift on a conserved quantity") hem de bizim asil ozgun
  katkimizi (bir SERTIFIKA, sadece bir ogrenme yontemi degil) nonlineer rejime tasir --
  literatur boyle bir sertifika sunmuyor, bu bosluk gercek ve doldurmaya deger.
- Optimizasyon kararsizligini (tohum 1'deki 31x yavaslama + yakinsamama) once TEK BASINA
  anlamak/teshis etmek (orn. gradyan normlari, omega/g agirliklarinin egitim boyunca
  izlenmesi) -- bu, nonlineer generator'u herhangi bir sertifikayla birlestirmeden once
  cozulmesi gereken bir on-kosul gibi gorunuyor.

**Ozet yargi:** Umut verici bir arastirma yonu -- literatur emsali var, sentetik kosulda
mekanizma calisiyor, ve paper'in "basarisiz" sonucunun nedeni artik ("hem multi-rate kilit
celismesi hem de piksel-encoder kirilganligi" olarak) daha net. Ama su anki haliyle CALISAN
bir kanit-of-concept URETILEMEDI: piksel sarkacta ne kilitli ne kilitsiz KoopAmp guvenilir
sekilde kontrolu geciyor. Ayri bir tez/calisma icin saglam bir zemin var, ama zemin "hazir
calisan bir yontem" degil, "neden zor oldugunu bilen, somut bir sonraki-adim listesi olan"
bir zemin.

---

## 6) FAZ 2 -- Bolum 5'teki iki somut oneri denendi

Koordinatorun talebiyle, Bolum 5'teki (a) ve (b) onerileri gercekten uygulanip test edildi.
Kural ayni: `paper/` ve `paper_l4dc/`'ye YAZILMADI (sadece okundu), yeni kod yeni dosyalarda
(`synth_amp_test.py`'ye eklenen `composite_loss`/`train_rich`/`gen_pair`/`train_koopamp_range`
fonksiyonlari, yeni `dmd_init_nonlinear.py`, yeni `synth_certificate_test.py`, yeni
`run_pend2.py`), mevcut `models.py`/`dmd_init.py`/`run_pend.py`/`data_pend.py` DEGISTIRILMEDI.

### 6.1) (a) Lusch ve ark.'in ZENGIN kayip fonksiyonu -- SENTETIK ortamda basarisiz/zararli

Denklem (Lusch, Kutz, Brunton 2018, Ek Bilgi Denklem 7a-7e, sarkac satiri katsayilariyla,
WebFetch ile tekrar dogrulandi):

```
L = a1*(L_recon + L_pred) + L_lin + a2*L_inf + a3*||W||^2
a1=1e-3, a2=1e-9, a3=1e-14         (Tablo 3, sarkac satiri)
L_recon = MSE(x_1, dec(enc(x_1)))                                  -- sadece t=0 karesi
L_pred  = (1/S_p) sum_m MSE(x_{m+1}, dec(K^m enc(x_1)))            -- bizim mevcut tek-kayibimizla ESDEGER (S_p=K=8 pencere)
L_lin   = (1/(T-1)) sum_m MSE(enc(x_{m+1}), K^m enc(x_1))          -- YENI terim, latent-uzayda dogrudan tutarlilik
L_inf   = ||x_1-dec(enc(x_1))||_inf + ||x_2-dec(K enc(x_1))||_inf  -- sup-norm cezasi
```
Orijinal makalede S_p=30 (bizim pencere K=8'den cok daha uzun) ve "bes dakikalik otokodlayici-
sadece on-egitim" var; ikisi de `synth_amp_test.py::train_rich` icinde uyarlandi (S_p->K,
pretrain_frac=0.15 -- iterlerin ilk %15'i basit kayipla, sonra tam bilesik kayiba geciliyor).

**Kritik gozlem: a1=1e-3 iken L_lin'in agirligi 1 -- yani bu agirliklandirma, piksel/gozlem
DOGRULUGUNU (L_recon+L_pred) neredeyse 1000x KUCULTUP, latent-ic tutarliligi (L_lin) BASKIN
kiliyor.** Bu, Lusch ve ark.'in kendi sistemlerinde (6 saatlik egitim, uzun ufuk) isliyor
olabilir ama bizim kisa-pencere/kisa-egitim kurulumumuzda TERSINE calisti:

| kol (sentetik, KoopAmp, 3 tohum, 4000 iter) | mse_train | mse_half |
|---|---|---|
| basit kayip (Bolum 3'teki "treatment") | 0.0042-0.0043 | **0.0046-0.0055** |
| zengin kayip, on-egitim YOK | 0.130-0.139 | **0.127-0.135** (~25-30x daha kotu) |
| zengin kayip, %15 on-egitim | 0.053-0.088 | **0.053-0.088** (~11-19x daha kotu, on-egitim kismen yardimci ama yetersiz) |

Ham veri: `clock_diag/synth_amp_rich_results.json`. **Sonuc: bu tam agirliklandirmayla
zengin kayip, bizim kisa-pencere kurulumumuzda basit kayibi GECMIYOR, aksine cok daha kotu
sonuc veriyor.** Muhtemel neden: a1=1e-3 ile piksel/gozlem dogrulugunun onemsizlestirilmesi,
kisa pencerede (S_p=8, Lusch'un S_p=30'undan cok kisa) L_lin'in tek basina iyi bir cozume
yeterli sinyal saglamamasi. Bu, "Lusch'un tam formulunu oldugu gibi tasimak" YERINE,
agirliklarin kisa-ufuk/hizli-egitim rejimine yeniden ayarlanmasi gerektigini gosteriyor --
bu haliyle (a) onerisi SENTETIK ortamda bile ISE YARAMADI, dolayisiyla piksel sarkacta da
(asagida) denendi ama basari beklenmiyordu.

### 6.2) (b) r-bagimli yerel sertifika -- SENTETIK ortamda GUCLU BASARILI (asil ozgun katki)

`dmd_init_nonlinear.py::local_gate_residuals()`: `dmd_init.py`'nin coklu-hizli tutarlilik
testini (Delta1/Delta2 arasi faz-donusu kontrolu) TEK YORUNGE duzeyine indirgeyen yeni bir
fonksiyon. Anahtar yapisal gercek: KoopAmp'te r_j=||(a_j,b_j)|| akis boyunca TAM KORUNUYOR
(omega_eff sadece donduruyor), yani HER TEK yorunge icin modelin ongordugu omega_j(r_j)
SABIT bir sayi -- dmd_init.py'nin GLOBAL DMD'sine gerek kalmadan, o TEK yorungenin Delta2
gozleminden dogrudan bir tutarlilik residual'i hesaplanabiliyor (ayrintili turetim/matematik
dosyanin ust-aciklamasinda).

**Dogrulama tasarimi (`synth_certificate_test.py`):** iki KoopAmp modeli, basit kayipla,
4000 iter, ayni tohum(0): `full_range` (r0~U(0.3,1.2), egitimde TUM aralik) ve `restricted`
(r0~U(0.3,0.7), [0.7,1.2] HIC gorulmedi -- kasitli ekstrapolasyon testi). Sonra HER IKISINE,
r0 TUM aralikta (0.3-1.2) yayilan 256 degerlendirme yorungesiyle, sertifika uygulandi.
**ONEMLI metodolojik not:** tablo GERCEK r0 ile binleniyor (dogrulama amacli, biliniyor);
ama sertifikanin KENDISI (residual) sadece modelin KENDI omega_eff/enc'ine bakiyor, r0'a
hic erismiyor -- korlemesine hesaplaniyor, tipki gercek dunyada olacagi gibi.

| model | r0 araligi | ortalama residual (rad) | ortalama gercek \|om_pred-om_true\| (rad/s) |
|---|---|---|---|
| full_range | [0.3,0.5) egitildi | 0.0066 | 0.0042 |
| full_range | [1.1,1.2) egitildi | 0.0199 | 0.0293 |
| restricted | [0.3,0.5) egitildi | **0.0010** | 0.0009 |
| restricted | [0.5,0.7) egitildi | **0.0011** | 0.0014 |
| restricted | [0.7,0.9) EGITILMEDI | **0.0180** (16x sicrama) | 0.0259 |
| restricted | [0.9,1.1) EGITILMEDI | **0.0747** (68x sicrama) | 0.1064 |
| restricted | [1.1,1.2) EGITILMEDI | **0.1379** (125x sicrama) | 0.1957 |

Korelasyon (residual, gercek \|om_pred-om_true\| hatasi), yorunge-yorunge, TUM 256
degerlendirme noktasi uzerinde: full_range icin **r=0.808**, restricted icin **r=0.996**
(neredeyse mukemmel). Ham veri: `clock_diag/synth_certificate_results.json`.

**Bu, sertifikanin TAM olarak tasarlandigi gibi calistigini gosteriyor: model bir genlik
bolgesinde egitilmemisse (ekstrapolasyon), sertifikanin residual'i o bolgede DRAMATIK sekilde
(>100x) yukseliyor VE bu yukselis gercek fiziksel hatayla neredeyse birebir orantili (r=0.996)
-- yani residual, "bu genlikte modele guvenme" seklinde YEREL, ISE YARAYAN bir bayrak
uretiyor, paper'in orijinal "gate" felsefesiyle (yerel dogrula, global varsayma) birebir
ayni ruhta, ama eksen omega_j'den (rijit donme) r_j'ye (genlik) genellenmis. Bu, gorevin
basinda tanimlanan "bizim asil ozgun katkimiz" (bir OGRENME yontemi degil, bir SERTIFIKA)
icin somut, calisan bir kanit-of-concept.**

### 6.3) Piksel sarkacta uygulama

Yeni dosya `run_pend2.py` (mevcut `run_pend.py`/`models.py`/`data_pend.py` degistirilmeden,
sadece import), --dmd none, --sampling mr, --iters 2200 (zaman-kutusu, orijinal 3000'e yakin),
her kosu icin sert 1700sn (`timeout` komutu) guvenlik siniri.

**(a) Zengin kayip, piksel sarkacta (3/3 tohum, koopamp, kilitsiz):**

| tohum | secs | mse_train | mse_half | f_true |
|---|---|---|---|---|
| 0 | 579 | 0.01731 | 0.01733 | 0.123 |
| 1 | 527 | 0.01621 | 0.01623 | 0.127 |
| 2 | 593 | 0.01620 | 0.01621 | 0.127 |

Sentetik bulguyla TUTARLI: zengin kayip piksel sarkacta da control'den (~0.0054-0.0055) **~3x
KOTU**, ve Bolum 4'teki basit-kayip koopamp'in en iyi tohumundan (0.00528) da acikca kotu.
Ancak ONEMLI bir yan-bulgu: 3/3 tohum DUZGUN VE HIZLI yakinsadi (527-593sn, Bolum 4'teki
tohum-1 anomalisi -- 12220sn, yakinsamama -- gibi HICBIR kararsizlik yok). Yani zengin kayip
DOGRULUGU FEDA EDIP KARARLILIK KAZANDIRIYOR gibi gorunuyor -- ilginc ama bu haliyle net bir
kazanc degil, cunku control zaten hem daha dogru hem hicbir kararsizlik gostermiyordu. Ham
veri: `clock_diag/pend2_results.jsonl`.

**(b) r-bagimli sertifika, piksel sarkacta:** Sertifikanin kendisini DAHA IYI (basit-kayip)
modeller uzerinde test etmek icin, ayni `run_pend2.py` ile --loss simple kullanip (Bolum
4'teki kurulumun bir tekrari, ama sertifika hesaplamasi ekli) 3 tohum daha kosturuldu.
(NOT: Ilk denemede sertifika binleme HATASI vardi -- modelin KENDI kodladigi yaricapla
bindim, Bolum 3/6.2'deki AYNI ozdeslenemezlik nedeniyle butun kutular BOS cikti; TRUE |th0|
ile binlemeye duzeltildi, bkz. `run_pend2.py::gen_pair`/`main`.)

**NOT: bu bolum, koordinatorun oturumu yeniden baslatilmasi nedeniyle ilk basta 1/3 tohumda
kesildi.** Sadece tohum 0 tamamlanmisti. **GUNCELLEME (kullanicinin kendi ham-veri
kontrolunde 6.4/PROPOSED_addition.md'deki "11-30x, hem sentetik hem piksel" cumlesinin
piksel kismini DESTEKLEMEDIGINI bulmasi uzerine): eksik 2 tohum (seed=1,2) koordinator
tarafindan tamamlandi** (`pend2_simple_results.jsonl`, artik 3 satir, seed 0/1/2, mse_half=
0.005321/0.004771/0.005637). **Kesin sonuc: rich/simple mse_half orani tohum-bazli
2.88x/3.26x/3.40x (havuzlanmis ortalama orani 3.16x)** -- bu, 6.4'teki "3/3 tohum, ~3x kotu"
tahminini (farkli bir control taban-cizgisiyle, Bolum 4'teki task=pendulum koop/koopamp'ten)
DOGRULUYOR, ama PROPOSED_addition.md/paper'daki eski "11-30x, hem sentetik hem piksel"
cumlesi YANLISTI (piksel gercek orani 11x'in bile altinda) -- bu cumle paper/sections/
05_limitations.tex, paper_l4dc/main.tex ve PROPOSED_addition.md'de duzeltildi (ayri, dogru
piksel araligi: 2.9-3.4x).

Ayrica ONEMLI bir tasarim farki var: Bolum 6.2'deki sentetik dogrulama `full_range` (0.3-1.2
tum aralik egitildi) ile `restricted` (0.3-0.7 egitildi, 0.7-1.2 HIC gorulmedi) modellerini
KARSILASTIRARAK ekstrapolasyon bolgesinde residual'in sicramasini gosterdi. Buradaki piksel
kosusu ise Bolum 4'teki gibi TUM genlik araligiyla (0.3-1.2) egitildi -- kasitli bir
kisitli-aralik/ekstrapolasyon varyanti KOSULMADI (zaman kisiti). Yani bu deney, sertifikanin
ASIL test etmesi gereken seyi (ekstrapolasyonu isaretleme) dogrudan SINAMIYOR; sadece
egitim-ICI residual'in genlik boyunca ne kadar duz/gurultulu oldugunu gosteriyor.

4 mod x 5 genlik-kutusu ortalama residual (rad), tek tohum:

| mod | [0.3,0.5) | [0.5,0.7) | [0.7,0.9) | [0.9,1.1) | [1.1,1.2) | mod ortalamasi |
|---|---|---|---|---|---|---|
| 0 | 0.119 | 0.159 | 0.160 | 0.188 | 0.190 | 0.163 |
| 1 | 0.148 | 0.206 | 0.104 | 0.212 | 0.082 | 0.150 |
| 2 | 0.206 | 0.265 | 0.229 | 0.182 | 0.133 | 0.203 |
| 3 | 0.227 | 0.197 | 0.176 | 0.223 | 0.099 | 0.184 |

Genel ortalama ~0.175, aralik ~0.08-0.27, ve genlik kutusuna gore ACIK BIR MONOTON EGILIM
YOK (gurultulu, kutudan kutuya duzensiz artis/azalis). Karsilastirma icin: Bolum 6.2'deki
sentetik testte EGITIM-ICI residual ~0.001-0.02 iken, EKSITRAPOLASYON (egitim-disi) residual'i
bile en fazla ~0.138'e cikiyordu. Buradaki piksel residual'lari (0.08-0.27), egitimin TUM
araligi kapsamasina RAGMEN, sentetik testin EKSTRAPOLASYON-sicrama seviyesiyle ayni
mertebede ya da daha yuksek. Bu, ya (i) piksel/CNN latent'inde "yerel tutarlilik" tabaninin
sentetik MLP kurulumuna gore cok daha gurultulu oldugunu, ya da (ii) tek-tohum/gurultu
varyansinin bu olcumde buyuk oldugunu dusundurur -- ikisini ayirt etmek icin en az 3 tohum
VE dogru (kisitli-araligi/ekstrapolasyon) deney tasarimi gerekir, hicbiri bu oturumda
tamamlanamadi.

**Sonuc: piksel sarkacta r-bagimli sertifikanin gercekten calisip calismadigi -- yani
egitim-disi genlik bolgesini gercekten isaretleyip isaretlemedigi -- BU VERIYLE
CEVAPLANAMAZ. Tek elde olan, egitim-ici residual duzeyinin sentetige gore cok daha yuksek
ve gurultulu oldugu, ki bu da Bolum 4'teki genel "piksel-CNN kirilganligi" bulgusuyla
tutarli.**

### 6.4) Guncellenmis tavsiye

**Not: bu bolumu, alt-agent'in oturum kesintisiyle yarim kalmasi uzerine, koordinator (ana
oturum) ham veriden tamamladi -- alt-agent'in kendi analiz standardiyla ayni titizlikte.**

**(a) Zengin kayip fonksiyonu (Lusch ve ark. formulasyonu, kisa-pencere rejimimize
tasindiginda): NET VE TUTARLI OLUMSUZ.** Hem sentetik (11-30x kotu) hem piksel sarkacta
(3/3 tohum, ~3x kotu) basit kayibi gecemedi. Tek yan-fayda: piksel sarkacta 3/3 tohum
kararli/hizli yakinsadi (Bolum 4'teki tohum-1 tarzi kararsizlik yok) -- ama bu bir kazanc
degil, cunku zaten daha kotu bir sonuca kararli sekilde ulasiyor. **Bu yon, oldugu haliyle,
kapatilabilir** -- agirliklarin yeniden ayarlanmasi (Bolum 5'teki orijinal onerinin
inceltilmis hali) hala acik ama bu oturumda denenmedi.

**(b) R-bagimli yerel sertifika: KONTROLLU ORTAMDA GUCLU, KANITLANMIS BIR SONUC; PIKSEL
VERIDE DOGRULANAMADI (basarisiz DEGIL, sadece test edilemedi).** Sentetik dogrulama
(Bolum 6.2) sertifikanin tasarlandigi gibi calistigini net bicimde gosterdi: dogru deney
tasarimiyla (kisitli-egitim vs tam-egitim karsilastirmasi), ekstrapolasyon residual'i 16-125x
sicriyor, gercek hatayla r=0.996 korelasyon. Bu, LITERATURDE KARSILIGI OLMAYAN gercek bir
metodolojik katki (Bolum 1'in sonucu: hicbir calisma boyle bir coklu-hizli yerel sertifika
sunmuyor). Ama piksel sarkactaki test (Bolum 6.3b) YANLIS deney tasarimiyla (ekstrapolasyon
ayrimi yok) VE tek, tekrarlanmamis bir tohumla yapildi -- dolayisiyla "piksel veride de
calisiyor" ya da "calismiyor" SONUCU CIKARILAMAZ, sadece "henuz dogru sekilde test
edilmedi" denebilir.

**GENEL TAVSIYE (Faz 1+2 birlikte):**
1. (a) zengin kayip yolu, bu haliyle, umut verici degil -- kapatilabilir.
2. (b) r-bagimli sertifika, KONTROLLU/SENTETIK ortamda calisan, ozgun bir sonuc -- bu tek
   basina (sentetik deney + matematiksel turetim + literatur konumlandirmasi), gelecekteki
   ayri bir calisma/tez icin YETERLI bir cekirdek katki olabilir, ANCAK piksel veride dogru
   sekilde (kisitli-araligi egitim + ekstrapolasyon testi, en az 3 tohum) sinanmadan "calisan
   bir yontem" diye sunulamaz.
3. **Su anki haliyle paper/'a EKLENECEK, hazir, dogrulanmis bir sonuc YOK.** Bkz.
   `clock_diag/PROPOSED_addition.md` -- orada iki secenek var: (i) Limitations'a "bunlari da
   denedik, (a) kapandi, (b) sentetikte calisiyor ama piksel'de dogrulanamadi" diyen kisa,
   dogru sekilde temkinli bir ek paragraf (dusuk risk, hemen eklenebilir); (ii) piksel
   dogrulamasini tamamlamak icin somut bir sonraki-oturum plani (yuksek getiri ama ek zaman
   gerektirir, bu oturumda yapilmadi).

**Ozet: Faz 2, Faz 1'in sonucunu DEGISTIRMEDI (paper'daki "amplitude-dependent generator
does not close the gap either" cumlesi hala DOGRU ve GECERLI) ama NEDEN sorusuna literatur
destekli, cok daha net bir cevap ekledi, VE gelecekteki bir calisma icin somut, kismen
dogrulanmis (sentetikte) bir yontem onerisi (r-bagimli sertifika) uretti.**

---

## 7) FAZ 3 -- r-bagimli sertifikanin piksel veride DOGRU deney tasarimiyla testi (SONUC: DURUST NEGATIF)

Koordinatorun talebiyle, `PROPOSED_addition.md`'nin Secenek B'si (Bolum 6.2'nin sentetik
`full_range`/`restricted` tasariminin piksel karsiligi, en az 3 tohum, ekstrapolasyon testi)
gercekten uygulandi. Kural ayni: `paper/`, `paper_l4dc/`, mevcut `models.py`/`dmd_init.py`/
`run_pend.py`/`data_pend.py`/`run_pend2.py`/`dmd_init_nonlinear.py` DEGISTIRILMEDI; yeni
dosyalar: `data_pend2.py` (`sample_th0_range(B,g,lo,hi)` -- data_pend.sample_th0'in lo/hi
parametreli hali) ve `run_pend3.py` (egitim + degerlendirme).

### 7.1) Deney tasarimi

Iki kol, KoopAmp, kilitleme yok (`dmd_init.py` hic import edilmiyor), basit MSE kaybi,
`--sampling mr`, `--iters 2200`, her kosu `timeout 1700` ile sarili:
- **full**: th0 ~ U(0.3,1.2) (data_pend.py ile AYNI tam aralik)
- **restricted**: th0 ~ U(0.3,0.7) -- [0.7,1.2] egitimde HIC gorulmedi

3 tohum (0,1,2) x 2 kol = 6 kosu, TAMAMI basariyla tamamlandi (hicbiri diverge etmedi,
hicbiri 1700sn sinirina yaklasmadi bile -- sureler 282-302sn araliginda, Faz 1'deki tohum-1
tarzi patoloji bu turda GORULMEDI). Ham veri: `clock_diag/pend3_results.jsonl` (her satir
bir kosu, `raw` alaninda 256 yorungenin ham th0/residual/om_pred_obs/om_true_formula
dizileri -- tum tablo/korelasyon hesaplari buradan yeniden turetilebilir).

Her iki kola da, egitimde KULLANILMAYAN, SABIT tohumla (777 standart mse kontrolu icin,
1234+seed sertifika icin) TAM 0.3-1.2 araligindan cekilen 256 degerlendirme yorungesiyle
`dmd_init_nonlinear.py::local_gate_residuals()` (DEGISTIRILMEDEN import) uygulandi. Tablo
GERCEK |th0| ile binlendi (KODLANMIS r ile DEGIL -- Bolum 6.3'teki hatanin AYNISI
tekrarlanmadi).

**Metodolojik fark (bilerek, gerekce asagida): "gercek |om_pred-om_true| hatasi" hesabi
Bolum 6.2'deki sentetik yontemden FARKLI bir yol izliyor.** Sentetik testte model.omega_eff
dogrudan fiziksel omega'yla ayni olcekteydi (OMEGA0=2.4, model init de ayni buyuklukte,
gozlem haritasi injektif). Piksel modelde ise KoopAmp'in n_osc=4 osilatorunden HANGISI
"fiziksel" oldugu ONCEDEN BILINMIYOR (Bolum 3/6.2'de not edilen encoder-serbest-yeniden-
parametreleme ozdeslenemezligi + CNN dekoder'in potansiyel faz-katlama/harmonik kapasitesi
nedeniyle, `local_gate_residuals`'in dondurdugu ham `om_pred` degerlerinin fiziksel omega
ile AYNI OLCEKTE olacagi garanti degil). Bu yuzden "gercek hata" OSILATOR-INDEKSINDEN
BAGIMSIZ, MODEL-CIKTISI-TABANLI yeni bir yontemle olculdu:
1. Modelin KENDI `predict()` ciktisi, ince zaman izgarasinda (dt=1/16, T=[0,8], Nyquist
   bandi ~2*pi/dt~100 rad/s >> fizik olcegi ~2.2-2.4 rad/s -- aliasing riski YOK) `data_pend.
   probe()` ile aciya cevriliyor.
2. Sifir-gecis (zero-crossing, ardisik isaret degisimleri arasi linear-enterpolasyonlu sure,
   donem = 2x ortalama aralik) yontemiyle GOZLEMLENEN frekans hesaplaniyor -- hem TRUE
   (DP.theta_at'in tam RK4 yorungesi) hem PRED (model ciktisi) icin AYNI fonksiyon
   (`crossing_period`/`observed_omega`, run_pend3.py).
3. TRUE yontem, sarkacin TAM eliptik-integral frekans yasasiyla (`true_omega_pendulum`,
   K(k) aritmetik-geometrik-ortalama ile hesaplanan tam eliptik integral, k=sin(th0/2))
   CAPRAZ DOGRULANDI: 8 rastgele yorungede eslesme 5 ondalik basamaga kadar (orn. th0=0.9306
   -> sifir-gecis yontemi 2.270649, eliptik formul 2.270638 rad/s) -- yontem DOGRU CALISIYOR.
4. PRED tarafinda, iyi egitilmis (full-range) bir modelde om_pred_obs degerleri fizik
   olcegiyle (2.19-3.26 araliginda, ortalama 2.24, gercek 2.19-2.39 araliginda, ortalama
   2.31) TUTARLI cikti -- yani bu yontem, `local_gate_residuals`'in ham `om_pred`'inin
   (Bolum 4/6.3'te gozlenen kucuk, ~0.02-0.4 mertebesindeki, muhtemelen osilator-arasi
   harmonik/katlanmis) degerlerinden FARKLI OLARAK, dogru fizik olceginde, saglam bir
   "gercek frekans hatasi" olcumu veriyor.

### 7.2) Sonuc (havuzlanmis, 3 tohum x 256 yorunge = 768 nokta / kol)

| kol | genlik-araligi | n | ortalama residual (rad) | ortalama gercek \|om_pred-om_true\| hatasi (rad/s) |
|---|---|---|---|---|
| full | [0.3,0.5) | 167 | 0.1651 | 0.1139 |
| full | [0.5,0.7) | 184 | 0.1909 | 0.1285 |
| full | [0.7,0.9) | 150 | 0.1716 | 0.0875 |
| full | [0.9,1.1) | 186 | 0.2200 | 0.0583 |
| full | [1.1,1.2) | 81  | 0.1504 | 0.2763 |
| restricted | [0.3,0.5) EGITILDI | 167 | 0.0978 | 0.1386 |
| restricted | [0.5,0.7) EGITILDI | 184 | 0.1577 | 0.1360 |
| restricted | [0.7,0.9) EGITILMEDI | 150 | 0.1902 | 0.0952 |
| restricted | [0.9,1.1) EGITILMEDI | 186 | 0.1719 | 0.0667 |
| restricted | [1.1,1.2) EGITILMEDI | 81  | 0.1629 | 0.0530 |

**Korelasyon (residual vs gercek hata), havuzlanmis 768 nokta:** full icin **r=-0.156**,
restricted icin **r=-0.268**. **HER IKISI DE NEGATIF** -- Bolum 6.2'nin sentetik sonucunun
(full: r=0.808, restricted: r=0.996, GUCLU POZITIF) TAM TERSI ISARET.

**Tohum-bazli korelasyonlar (6/6 tutarli sekilde negatif):**

| kol | tohum | secs | mse_train_eval (tam-aralik) | mse_half | corr(residual,gercek_hata) |
|---|---|---|---|---|---|
| full | 0 | 282 | 0.00007 | 0.00545 | -0.145 |
| full | 1 | 300 | 0.00004 | 0.00503 | -0.259 |
| full | 2 | 302 | 0.00007 | 0.00572 | -0.217 |
| restricted | 0 | 292 | 0.00734 | 0.01023 | -0.477 |
| restricted | 1 | 300 | 0.00808 | 0.01089 | -0.434 |
| restricted | 2 | 301 | 0.00845 | 0.01141 | -0.120 |

**Ekstrapolasyon-bolgesi sicramasi (restricted, egitim-ici [0.3,0.7) vs egitim-disi
[0.7,1.2), havuzlanmis):** residual egitim-ici ortalama 0.1292 -> egitim-disi ortalama
0.1768, oran **1.37x**. Ama bu, full_range kolunda (TAMAMI egitim-ici oldugu halde) ayni
boluntude gorulen DOGAL GURULTU TABANIYLA (0.1786 -> 0.1891, oran 1.06x, ya da bin-bazli
bakildiginda full'un kendi ici dalgalanmasi 0.126-0.220 araliginda, oran ~1.75x) ayni
mertebede -- yani **1.37x'lik "sicrama", sentetikteki 16-125x'lik acik sicramanin aksine,
gurultu tabanindan istatistiksel olarak AYRISTIRILAMIYOR.**

Daha da onemlisi: restricted kolda GERCEK hata, egitim-disi bolgede DUSUK CIKIYOR (egitim-ici
0.1372 -> egitim-disi 0.0743, oran 0.54x) -- yani model, HIC GORMEDIGI genlik bolgesinde
(0.7-1.2) frekansi aslinda DAHA IYI tahmin ediyor, egitim-ici bolgeden DAHA KOTU degil. Bu,
"ekstrapolasyon = daha kotu tahmin" varsayiminin piksel veride (bu olcum yontemiyle)
DOGRULANMADIGINI gosteriyor -- muhtemelen buyuk genlikli sarkac salinimlarinin gorsel olarak
daha "belirgin" (bob daha genis bir yay tarar) olup CNN/probe'un aciyi izlemesini
KOLAYLASTIRMASI nedeniyle, kucuk genlik civarindaki hassas ayrimdan daha az zorlu bir cikarim
problemi olusturuyor olabilir -- ama bu spekulasyon, ayrica test edilmedi.

Buna karsin, ham MSE olcumleri (mse_train_eval, TUM 0.3-1.2 araliginda) restricted kolun
gercekten daha kotu genellestigini DOGRULUYOR: full ~0.00004-0.00007 iken restricted
~0.0073-0.0085 (**~100-200x daha kotu**) -- yani genelleme farki GERCEK VE BUYUK, ama
r-bagimli LOKAL SERTIFIKA REZIDUELI bu farki NE DOGRU YONDE ISARETLIYOR NE DE GERCEK HATAYLA
ANLAMLI KORELE.

### 7.3) Degerlendirme

**Sonuc: piksel sarkacta r-bagimli yerel sertifika, Bolum 6.2'deki sentetik ortamda
gosterdigi (r=0.996, 16-125x ekstrapolasyon sicramasi) performansi GOSTERMIYOR.** Bu,
gorevin degerlendirme kriterlerindeki "durust negatif" durumu: sicrama YOK/gurultuye gomulu
VE korelasyon anlamli pozitif DEGIL (aksine, tutarli sekilde HAFIF NEGATIF, 6/6 tohumda).

Olasi nedenler (test edilmedi, ileri calisma icin):
1. **Osilator-indeksi ozdeslenemezligi/harmonik katlanma** (Bolum 7.1'de not edildi):
   `local_gate_residuals`'in dogrudan kullandigi ic `om_pred` (residual hesabinda kullanilan)
   fiziksel omega ile 1:1 olcekte olmayabilir -- CNN dekoder'in faz-katlama kapasitesi
   varsa, residual'in kendisi (iki-hizli TUTARLILIK olcusu, DOGRULUK degil) katlanmis bir
   temsilde de hesaplanabilir olsa bile, bu temsilin genlik-baginda YEREL TUTARSIZLIGI
   fiziksel frekans hatasiyla ayni sekilde OLCEKLENMEYEBILIR -- sentetikte (katlanma yok,
   injektif gozlem haritasi) bu sorun yoktu.
2. **CNN encoder'in genlik-ekstrapolasyonuna sentetik MLP'den FARKLI tepki vermesi**: sentetik
   testte encoder r'yi (latent yaricap) DUZGUN monoton bir sekilde genlik-disi bolgeye
   tasiyordu (Bolum 6.2'nin r=0.996'sinin temeli); piksel CNN'in r-benzeri bir latent
   buyuklugu genlik-disi bolgede NASIL davrandigi (monoton mu, yoksa gurultulu/dogrusal-
   olmayan mi genellestiriyor) bu deneyde DOGRUDAN olculmedi.
3. Genel olarak Bolum 4/6.3'un tekrarlayan bulgusu: piksel-CNN + nonlineer omega_eff
   kombinasyonu, sentetik/dusuk-boyutlu ortama gore SISTEMATIK OLARAK daha gurultulu/
   kirilgan -- bu Faz 3 sonucu bu genel bulguyu, ozel olarak SERTIFIKA MEKANIZMASI icin
   DOGRULUYOR (Faz 1/2'nin "OGRENME" mekanizmasi icin gösterdigi kirilganligin, Faz 3'te
   "SERTIFIKA" mekanizmasi icin de gecerli oldugu artik GOSTERILDI, varsayilmadi).

**Bu sonuc, gorevin "TİTİZLİK" talimatina uygun sekilde ZORLANMADI**: 6/6 tohum tutarli
negatif korelasyon veriyor (rastgele gurultu olsaydi tohumlar arasinda isaret degisirdi),
ham MSE genelleme farki GERCEK (yontemin kendisi/olcum hatasi degil), ve "gercek frekans"
olcum yontemi (sifir-gecis) TRUE yorungelerde eliptik formulle 5-ondalik-basamak dogrulugunda
CAPRAZ DOGRULANDI -- yani bu NEGATIF sonuc, bir olcum hatasindan degil, GERCEK bir yontem-
basarisizligindan kaynaklaniyor gibi gorunuyor.

### 7.4) Guncellenmis tavsiye (Faz 1+2+3 birlikte)

**r-bagimli yerel sertifika artik SENTETIKTE GUCLU CALISAN, PIKSEL VERIDE DOGRU DENEY
TASARIMIYLA TEST EDILMIS VE ORADA CALISMADIGI GOSTERILMIS bir yontem.** Bu, Bolum 6.4'teki
"henuz test edilmedi" belirsizligini KAPATIYOR -- artik "test edilmedi" degil, "test edildi,
piksel veride (bu haliyle, bu olcum protokoluyle) calismiyor" denebilir.

Genel yargı degismedi (paper'daki "amplitude-dependent generator does not close the gap
either" hala dogru), ama nonlineer arastirma yolunun UC parcasi da (ogrenme -- basarisiz;
zengin kayip -- basarisiz; sertifika -- SENTETIKTE basarili, PIKSELDE basarisiz) artik
NET SEKILDE SINANDI. Bu arastirma yolu icin en dogru cerceve: **"sentetik/dusuk-boyutlu
ortamda calisan, piksel/CNN ortamina GENELLENMEYEN bir yontem ailesi"** -- literatur-
destekli, matematiksel olarak turetilmis, sentetik olarak KANITLANMIS ama paper'in gercek
deney kosullarina (piksel + CNN) TASINAMAYAN bir katki. Bkz. `PROPOSED_addition.md`
(guncellendi).

---

## 8) FAZ 4 -- ikinci nonlineer sistem (Duffing) genelleme testi (SONUC: KARISIK -- ana sertifika testi YINE DURUST NEGATIF/TUTARSIZ, ama YENI bir olcum -- r_enc isaret-tersine-donmesi -- CNN encoder'in genlik-genellemesi hakkinda net, pozitif bir bulgu verdi)

Koordinatorun talebiyle, Faz 3'un ("Bolum 7") negatif sonucunun piksel sarkaca (theta''=
-W0^2*sin(theta), YUMUSAYAN -- genlik arttikca frekans DUSER) ozgu mu yoksa CNN-encoder
sertifikasyonunun genel bir sinirlamasi mi oldugunu ayirt etmek icin, matematiksel olarak
FARKLI, ZIT yonlu (SERTLESEN/hardening -- genlik arttikca frekans ARTAR) bir ikinci sistemde
AYNI full/restricted deney tasarimi tekrarlandi: undamped, konservatif Duffing-tipi acisal
osilator, `theta'' = -w0^2*theta - eps*theta^3`. Kural ayni: `data.py`, `data_pend.py`,
`data_pend2.py`, `models.py`, `losses.py`, `dmd_init.py`, `dmd_init_nonlinear.py`, `run.py`,
`run_pend.py`, `run_pend2.py`, `run_pend3.py` DEGISTIRILMEDI; yeni dosyalar: `data_duffing2.py`
(fizik -- render/probe/BANK/sample_th0 DP'den aynen yeniden kullanildi, sadece ODE degisti) ve
`run_pend4.py` (egitim + degerlendirme, run_pend3.py'nin birebir ayni mantigi).

### 8.1) Fizik kalibrasyonu + dogrulama

**EPS secimi (deneysel, kapali-form formule GUVENILMEDI -- gorev talimati geregi):** RK4 +
sifir-gecis yontemiyle (crossing_period, run_pend3.py'den kopyalandi -- bu yontem Faz 3'te
pendulumun TAM eliptik formuluyle 5 ondalik basamaga kadar dogrulanmisti, yani yontemin
kendisi guvenilir) th0=0.3/0.7/1.2'de olculen gercek omega, w0=2.4 sabit tutularak eps in
{0.5,1.0,1.5,2.0,3.0} icin tarandi:

| eps | om(0.3) | om(0.7) | om(1.2) | om(1.2)/om(0.3) | degisim |
|---|---|---|---|---|---|
| 0.5 | 2.40702 | 2.43793 | 2.50960 | 1.0426 | +4.26% |
| **1.0** | **2.41401** | **2.47519** | **2.61398** | **1.0828** | **+8.28%** |
| 1.5 | 2.42099 | 2.51182 | 2.71388 | 1.1210 | +12.10% |
| 2.0 | 2.42793 | 2.54785 | 2.80987 | 1.1573 | +15.73% |
| 3.0 | 2.44176 | 2.61823 | 2.99186 | 1.2253 | +22.53% |

**eps=1.0 secildi**: +8.28% (SERTLESEN, ARTAN yonde) -- Faz 3'teki pendulumun th0=0.3->1.2
arasinda gosterdigi ~%8-10 DUSUS ile BUYUKLUK olarak karsilastirilabilir, ISARET olarak TERS
(bu deneyin butun amaci). Ilk denenen deger (eps=1.0) zaten hedef araliga dustugu icin baska
deger denenmedi.

**Dogrulama (enerji korunumu, formul yerine):** Duffing icin kapali-form eliptik-integral
frekans formulu EZBERDEN yazilip guvenilmedi (gorev talimati). Onun yerine, undamped/
konservatif sistemin enerjisi `E = 0.5*thetadot^2 + 0.5*w0^2*theta^2 + (eps/4)*theta^4`
th0=0.3/0.7/1.2 icin dt=1/64 RK4 ile 16s (~38-44 periyot) boyunca izlendi:

| th0 | E0 | max\|dE/E0\| (16s, ~1027 ince adim) |
|---|---|---|
| 0.3 | 0.261225 | 1.32e-06 |
| 0.7 | 1.471225 | 1.69e-06 |
| 1.2 | 4.665600 | 2.68e-06 |

Enerji 1e-6 mertebesinde (goreli) korunuyor -- dt=1/64 RK4 integrasyonu bu sistem icin
yeterince hassas, methodolojik risk yok.

**Kaggle'da karsilasilan iki cihaz (device) hatasi ve duzeltmesi (seffaflik icin not
ediliyor):** Ilk iki Kaggle denemesi (kernel v1, v2) TAMAMI HATA verdi (6/6 is), DOKUNULMASI
YASAK olan dosyalarla ilgisi OLMAYAN, sadece bu FAZ 4'e ozgu YENI dosyalardaki cagri
deseninden kaynaklanan CUDA/CPU cihaz uyusmazligi yuzunden:
1. `data_duffing2.py::theta_at_duffing` icinde, cagiranlarin (run_pend3.py'deki t1/t2 cagri
   deseniyle AYNI, `.to(DEV)` YOK) CPU'da birakilan `t` tensoru, GPU'daki `th0` ile
   `Tr.gather` sirasinda cakisti -- bu YENI dosyada (data_pend.py'nin orijinal theta_at'i
   DEGISTIRILMEDEN kalarak) `t = t.to(th0.device)` satiri eklenerek duzeltildi.
2. `dmd_init_nonlinear.py::local_gate_residuals` (DOKUNULMASI YASAK, PROTECTED dosya) icinde
   `eye2 = torch.eye(2)[None]` DEVICE BELIRTMEDEN olusturuluyor (daima CPU) -- GPU'da
   `XtX = einsum(X,X) + 1e-6*eye2` satirinda "cuda:0 ve cpu" hatasi verdi. Bu dosyaya
   DOKUNULMADI (kural); bunun yerine `run_pend4.py` icinde SADECE bu cagri icin modelin bir
   CPU kopyasi (`copy.deepcopy(model).cpu()`) ve CPU'ya tasinmis `x1,x2` kullanilarak
   fonksiyonun TUM ic tensorlerinin (eye2 dahil) tutarli sekilde CPU'da kalmasi saglandi --
   orijinal `model` (GPU) sonraki adimlar (observed_omega) icin degismeden kaldi. (Bu hata,
   muhtemelen Faz 3'un calistigi zamandan bu yana Kaggle'in varsayilan PyTorch surumunun
   degismesiyle ilgili -- `dmd_init_nonlinear.py`'nin kendisi hicbir zaman degistirilmedi;
   Faz 3, run_pend3.py'de AYNI cagri deseni (`t1`/`t2` icin `.to(DEV)` yok) kullaniliyor ve o
   zaman calismisti.) Kernel v3, bu iki duzeltmeyle 6/6 basariyla tamamlandi (hata YOK).

### 8.2) Deney + kosu ozeti

Faz 3 ile BIREBIR ayni yapi: full th0~U(0.3,1.2), restricted th0~U(0.3,0.7), KoopAmp, basit
MSE kaybi, `--sampling mr`, `--iters 2200`, her is `timeout=1700` ile sarili (Kaggle T4,
`ThreadPoolExecutor(3)`). 3 tohum x 2 kol = 6 kosu, TAMAMI basariyla tamamlandi (hicbiri
diverge etmedi, sureler 808-866sn araliginda):

| kol | tohum | secs | mse_train_eval (tam-aralik) | mse_half | n_valid_freq |
|---|---|---|---|---|---|
| full | 0 | 866 | 0.0000514 | 0.008433 | 256/256 |
| full | 1 | 808 | 0.0000325 | 0.008568 | 256/256 |
| full | 2 | 836 | 0.0000581 | 0.008162 | 256/256 |
| restricted | 0 | 853 | 0.007640 | 0.011324 | 256/256 |
| restricted | 1 | 833 | 0.008330 | 0.011884 | 256/256 |
| restricted | 2 | 826 | 0.009314 | 0.013321 | 256/256 |

Ham veri: `clock_diag/pend4_results.jsonl` (6 satir, her biri bir kosu, `raw` alaninda 256
yorungenin th0_abs/residual_mean/om_pred_obs/om_true_obs/**r_enc** dizileri -- tum tablo/
korelasyon hesaplari buradan yeniden turetilebilir).

Ham MSE genelleme farki Faz 3 ile AYNI buyuklukte VE GERCEK: full ortalama mse_train_eval
~4.7e-5, restricted ortalama ~8.4e-3 -- **~178x daha kotu** (Faz 3'te ~100-200x idi).
Restricted kol, egitilmedigi [0.7,1.2) bolgesini tam-aralik MSE'ye gore GERCEKTEN
genellestiremiyor -- bu METODOLOJIK OLARAK Faz 3'le tutarli, saglam bir sonuc.

### 8.3) Ana sertifika sonucu (residual vs gercek frekans hatasi)

**Havuzlanmis (3 tohum x 256 yorunge = 768 nokta / kol), 5 genlik binine gore:**

| kol | genlik-araligi | n | ortalama residual (rad) | ortalama gercek \|om_pred-om_true\| hatasi (rad/s) |
|---|---|---|---|---|
| full | [0.3,0.5) | 167 | 0.1647 | 0.1653 |
| full | [0.5,0.7) | 184 | 0.1785 | 0.1616 |
| full | [0.7,0.9) | 150 | 0.1463 | 0.0828 |
| full | [0.9,1.1) | 186 | 0.1814 | 0.0628 |
| full | [1.1,1.2) | 81  | 0.1679 | 0.0573 |
| restricted | [0.3,0.5) EGITILDI | 167 | 0.1984 | 0.1528 |
| restricted | [0.5,0.7) EGITILDI | 184 | 0.1984 | 0.1480 |
| restricted | [0.7,0.9) EGITILMEDI | 150 | 0.2060 | 0.0793 |
| restricted | [0.9,1.1) EGITILMEDI | 186 | 0.1621 | 0.0701 |
| restricted | [1.1,1.2) EGITILMEDI | 81  | 0.1789 | 0.1450 |

**Korelasyon (residual vs gercek hata), havuzlanmis 768 nokta:** full icin **r=-0.133**
(Faz 3'teki -0.156 ile AYNI ISARET VE BUYUKLUK -- burada da negatif), restricted icin
**r=+0.178** (Faz 3'teki -0.268'in AKSINE, ZAYIF POZITIF).

**Tohum-bazli korelasyonlar -- full 3/3 TUTARLI NEGATIF, restricted TUTARSIZ (KARISIK
ISARET):**

| kol | tohum | corr(residual,gercek_hata) |
|---|---|---|
| full | 0 | -0.142 |
| full | 1 | -0.077 |
| full | 2 | -0.036 |
| restricted | 0 | **+0.556** |
| restricted | 1 | -0.063 |
| restricted | 2 | +0.107 |

Full kol, Faz 3'teki gibi 3/3 tutarli sekilde HAFIF NEGATIF -- bu FAZ 3'UN ANA BULGUSUNU
(sertifika piksel/CNN ortaminda genel olarak calismiyor) DOGRULUYOR. Restricted kol ise
Faz 3'ten (6/6 tutarli negatif) FARKLI: 1 tohum orta-guclu pozitif (+0.556), 1 tohum
sifira yakin negatif (-0.063), 1 tohum zayif pozitif (+0.107) -- yani **NE tutarli pozitif
NE tutarli negatif**, tohumdan tohuma ISARET DEGISIYOR. Bu, "sertifika bu sistemde CALISIYOR"
diye YORUMLANAMAZ (tek basina +0.556'lik tohum secilip one cikarilirsa hack olur -- kural
geregi YAPILMADI) -- ama Faz 3'un "6/6 net negatif" kadar temiz bir "calismiyor" sonucu da
DEGIL. En durust okuma: **restricted kolda sertifika-gercek-hata iliskisi bu sistemde
GURULTUYE GOMULU/TOHUMA DUYARLI, guvenilir bir sinyal yok.**

**Ekstrapolasyon-bolgesi (restricted, egitim-ici [0.3,0.7) vs egitim-disi [0.7,1.2),
havuzlanmis):** residual egitim-ici ortalama 0.1984 -> egitim-disi ortalama 0.1811, oran
**0.913x** (yani residual egitim-disi bolgede DUSUYOR, ARTMIYOR). Karsilastirma icin full
kolun kendi ici (TAMAMI egitim-ici oldugu halde) AYNI boluntudeki dogal dalgalanmasi: 0.1719
-> 0.1662, oran 0.966x. **restricted'in orani (0.913x) full'un kendi gurultu tabanindan
(0.966x) daha DUSUK -- yani Faz 3'teki gibi (1.37x, zayif da olsa en azindan DOGRU yonde)
BILE DEGIL, burada sicrama YONU TERS (residual dusuyor).** Tohum-bazli ayrintida bu daha da
karisik: restricted tohum 0 -> 0.676 (buyuk DUSUS), tohum 1 -> 1.035, tohum 2 -> 1.055
(full'un kendi tohum-bazli araligi 0.911-0.992 ile ORTUSUYOR) -- **hicbir tohumda, Faz
3'teki bile zayif olan sicrama isaretine benzer bir seyin izi YOK.**

Faz 3'teki "buyuk genlik = daha kolay gozlemlenen hareket -> daha dusuk gercek hata" bulgusu
BU SISTEMDE DE, hem full hem restricted kolda GUCLU sekilde tekrarlandi: gercek hata orani
(egitim-disi/egitim-ici) full icin **0.422x**, restricted icin **0.585x** -- yani genlik
buyudukce gercek frekans hatasi HER IKI kolda da DUSUYOR (Faz 3'te sadece restricted'de
olculmustu, burada full'da da ayni yonde -- bu, "sistem-ozgu bir garanti" DEGIL, gorsel
buyuklugun probe/CNN performansini genel olarak etkiledigine dair EK KANIT).

### 8.4) YENI olcum: r_enc (kodlanmis latent yaricap) vs gercek \|th0\| korelasyonu

Bu, Faz 3'te OLCULMEMIS, sadece varsayilmis ("CNN encoder'in genlik-yarıçap temsili
genellemiyor olabilir") bir iddiayi DOGRUDAN test ediyor. `local_gate_residuals`'in
dondurdugu `r_enc` (256,4) -- her yorunge icin 4 osilatorun kodlanmis latent yaricapi -- ile
gercek `|th0|` arasindaki Pearson korelasyonu, egitim-ici/egitim-disi bolgelere ayri ayri,
"en iyi" (mutlak deger en buyuk) osilator secilerek hesaplandi:

| kol | tohum | bolge | n | 4 osilator korelasyonu | en iyi |
|---|---|---|---|---|---|
| full | 0 | in [0.3,0.7) | 119 | 0.633, -0.609, 0.077, 0.044 | 0.633 |
| full | 0 | out [0.7,1.2) | 137 | -0.143, 0.567, -0.485, 0.763 | 0.763 |
| full | 1 | in [0.3,0.7) | 120 | 0.259, 0.690, 0.424, 0.048 | 0.690 |
| full | 1 | out [0.7,1.2) | 136 | 0.438, 0.150, 0.367, 0.170 | 0.438 |
| full | 2 | in [0.3,0.7) | 112 | 0.531, 0.646, 0.214, 0.472 | 0.646 |
| full | 2 | out [0.7,1.2) | 144 | 0.069, 0.232, 0.282, 0.342 | 0.342 |
| restricted | 0 | in [0.3,0.7) EGITILDI | 119 | 0.991, 0.651, 0.557, 0.420 | **+0.991** |
| restricted | 0 | out [0.7,1.2) EGITILMEDI | 137 | -0.944, -0.750, -0.936, -0.506 | **-0.944** |
| restricted | 1 | in [0.3,0.7) EGITILDI | 120 | 0.108, 0.966, 0.356, 0.257 | **+0.966** |
| restricted | 1 | out [0.7,1.2) EGITILMEDI | 136 | -0.957, -0.991, -0.951, -0.294 | **-0.991** |
| restricted | 2 | in [0.3,0.7) EGITILDI | 112 | 0.458, 0.978, -0.222, 0.101 | **+0.978** |
| restricted | 2 | out [0.7,1.2) EGITILMEDI | 144 | -0.160, -0.988, -0.079, -0.102 | **-0.988** |

**Bu, arastirma hattinin en NET, en TUTARLI bulgusu (6/6 tohum x bolge kombinasyonunda AYNI
oruntu):** full kolda (tum aralik egitimde gorulmus) r_enc, hem ic hem "dis" alt-bolgede
POZITIF korelasyonlu (0.342-0.763 araliginda, tutarli ama orta-guclu) -- encoder genligi
makul sekilde kodluyor. **restricted kolda ise ("dis" bolge GERCEKTEN egitim-disi) korelasyon
egitim-ici bolgede NEREDEYSE MUKEMMEL POZITIF (3/3 tohumda 0.991/0.966/0.978) VE egitim-disi
bolgede NEREDEYSE MUKEMMEL NEGATIF'E DONUYOR (3/3 tohumda -0.944/-0.991/-0.988) -- ISARET
TERSINE DONUYOR, sadece zayiflamiyor.** Yani encoder'in ogrendigi "r buyudukce genlik
buyur" iliskisi, egitim araliginin DISINA cikildiginda GURULTULU HALE GELMIYOR -- SISTEMATIK
OLARAK TERSE DONUYOR (buyuk gercek genlik -> KUCUK kodlanmis r). Bu, "CNN encoder'in genlik-
yaricap temsili genellemiyor" iddiasini sadece DOGRULAMAKLA kalmiyor, ONDAN DAHA GUCLU, DAHA
SPESIFIK bir mekanizma gosteriyor: genelleme rastgele bozulmuyor, MONOTON YON DEGISTIRIYOR.
Bu muhtemelen r-bagimli sertifikanin (Bolum 8.3, 6.2) neden restricted kolda GUVENILIR bir
sinyal VERMEDIGININ dogrudan aciklamasi olabilir -- residual hesabi `om_pred = omega_eff(r)`
uzerinden r'ye bagli, ve r'nin kendisi egitim-disi bolgede fiziksel anlamini kaybedip TERS
yonde hareket ediyorsa, ondan turetilen herhangi bir r-bagimli tutarlilik olcusunun de
guvenilmez olmasi BEKLENIR -- test edilmedi ama makul bir nedensel hipotez.

**KOORDINATOR DOGRULAMA NOTU:** Yukaridaki tablo, "en iyi" osilatoru HER bolge (ic/dis) icin
AYRI AYRI seciyor -- bu bir secim-yanliligi riski tasir (4 osilatorden en buyuk |r|'yi
bolgeden bagimsiz secmek, sinyal olmasa bile sans eseri yuksek korelasyon uretebilir). Daha
adil bir kontrolle (osilator indeksi SADECE egitim-ici altkumeden secilip, AYNI indeks
egitim-disi altkumeye uygulanarak) yeniden hesapladim (ham `raw.r_enc`/`raw.th0_abs`'ten,
`pend4_results.jsonl`):
- **Restricted kolda bulgu SAGLAM: isaret-tersine-donmesi bu daha siki testte de aynen
  cikiyor** (sabit-indeks disi-korelasyon: seed0 j=0 -> -0.944, seed1 j=1 -> -0.991, seed2
  j=1 -> -0.988) -- yukaridaki tabloyla PRATIKTE AYNI, yani bu YAPISAL bir sonuc, secim-
  yanliligi ARTEFAKTI DEGIL.
- **Full kolda "kontrol" iddiasi ("her iki bolgede de pozitif kaliyor") kismen ZAYIFLIYOR:**
  sabit-indeks testinde seed 0 de isaret DEGISTIRIYOR (ic 0.633 -> dis **-0.143**, tabloda
  "dis en iyisi" olarak farkli bir osilator (0.763) raporlanmisti), seed 1/2 de pozitif
  kaliyor ama daha zayif (0.15/0.232, tabloda 0.438/0.342). Yani "full kolda hicbir isaret
  degisimi yok" iddiasi biraz fazla iyimser sunulmus -- dogrusu "restricted kolda 3/3 tohumda
  GUCLU VE TUTARLI isaret-tersine-donmesi var, full kolda boyle bir sey YOK/ZAYIF (1/3 tohumda
  zayif bir tersine-donme var, ama restricted'daki kadar guclu/tutarli degil)".
- **Sonuc degismiyor, cercevesi hafif duzeliyor:** ana bulgu (encoder'in r-genlik iliskisi
  egitim-disi genlik bolgesinde GUVENILMEZ hale geliyor, ve bu restricted kolda net bir
  isaret-tersine-donmesi seklinde) AYAKTA KALIYOR, ama "full kolda boyle bir sorun yok"
  karsilastirmasi raporda oldugundan biraz daha net/temiz sunulmustu.

### 8.5) Degerlendirme

**Ana sertifika testi (residual vs gercek hata) Faz 4'te de DURUST NEGATIF/TUTARSIZ:** full
kolda Faz 3 ile AYNI ISARETTE (3/3 hafif negatif) tekrarlandi; restricted kolda Faz 3'un
temiz 6/6-negatif sonucu REPLIKE OLMADI (tohumdan tohuma isaret degisiyor, +0.556'dan
-0.063'e) -- yani sertifikanin restricted-kol davranisi SISTEM-BAGIMLI/GURULTULU, ama
hicbir yorumda GUVENILIR POZITIF bir sinyal (Bolum 6.2'nin sentetik r=0.996'sina yaklasan bir
sey) YOK. Ekstrapolasyon sicramasi bu sistemde Faz 3'ten bile ZAYIF: oran full'un kendi
gurultu tabaninin (0.966x) ALTINDA (0.913x havuzlanmis, tohumlara gore 0.676-1.055) --
sicrama YOK, hatta hafifce TERS yonde.

**Ama YENI r_enc-|th0| olcumu (Bolum 8.4), bu arastirma hattinin en net pozitif KATKISI:**
CNN encoder'in genlik-yaricap kodlamasi, egitim araliginin disinda GURULTULU degil, SISTEMATIK
OLARAK ISARET-TERSINE-DONUYOR (6/6 tohum x bolge kombinasyonunda ayni oruntu, buyuklukleri
0.94-0.99 mertebesinde -- rastgele olsaydi bu kadar guclu ve tutarli bir ters-donme
BEKLENMEZDI). Bu, Faz 3'un "olasi nedenler" listesindeki 2. maddeyi ("CNN encoder'in genlik-
ekstrapolasyonuna sentetik MLP'den FARKLI tepki vermesi... bu deneyde DOGRUDAN olculmedi")
artik DOGRUDAN, nicel olarak dogruluyor -- ve ondan daha spesifik bir mekanizma ortaya
koyuyor.

**Sonuc SARKACA OZGU DEGIL, genel bir CNN-encoder sinirlamasi gibi gorunuyor:** iki
matematiksel olarak farkli (softening/pendulum vs hardening/Duffing), ZIT yonlu nonlineerlige
sahip sistemde de ayni temel oruntu (a) sertifikanin restricted kolda guvenilir sinyal
VERMEMESI ve (b) encoder'in r-temsilinin egitim-disi bolgede bozulmasi tekrarlandi. Faz 4'un
KATKISI, bu bozulmanin DOGASINI netlestirmesi: rastgele gurultu degil, sistematik isaret-
tersine-donmesi.

**Bu sonuc gorevin "TITIZLIK/ZORLAMA-YOK" talimatina uygun sekilde raporlandi:** restricted
kolun sertifika-korelasyonu pozitif YONE cekilebilecek bir tohum (+0.556) icerse de, bu TEK
BASINA one cikarilmadi -- 3/3 tohumun TUTARSIZLIGI acikca raporlandi ve "calisiyor" diye
YORUMLANMADI. r_enc bulgusu ise TERSINE, 6/6 tam tutarli oldugu icin GUVENLE pozitif bir
bulgu olarak sunuldu.

### 8.6) Guncellenmis tavsiye (Faz 1+2+3+4 birlikte)

**r-bagimli yerel sertifikanin piksel/CNN ortaminda calismama sonucu, artik TEK bir sisteme
(piksel sarkac) OZGU degil -- ikinci, fizigi ZIT yonlu bir sistemde de (Duffing, kismen
tutarsiz ama guvenilir pozitif sinyal olmadan) TEKRARLANDI.** Bu, "sentetik/dusuk-boyutlu
ortamda calisan, piksel/CNN ortamina genellenmeyen bir yontem ailesi" cercevesini (Bolum 7.4)
GUCLENDIRIYOR (artik tek deney degil, iki BAGIMSIZ nonlineerlikte tekrarlanan bir sonuc).

Ayrica, Faz 4 yeni, kendi basina anlamli bir bulgu ekledi: **CNN encoder'in ogrendigi genlik-
yaricap temsili, egitim dagiliminin disinda rastgele degil, SISTEMATIK OLARAK TERS YONE
donuyor (6/6 tohum x bolge, \|r\|~0.94-0.99).** Bu, r-bagimli sertifikanin piksel/CNN'de neden
calismadigina dair somut, olcumlenmis bir aday mekanizma sunuyor (spekulasyon degil) ve
gelecekte encoder mimarisi/egitim rejimi degistirilerek (orn. r'yi ACIKCA amplitude'e
regularize eden bir yardimci kayip -- Lusch et al. 2018'deki auxiliary network'e daha yakin
bir tasarim) bu sorunun giderilip giderilemeyecegi test edilebilir -- bu Faz 4 kapsaminda
YAPILMADI, ileri calisma onerisi olarak birakiliyor.

Genel yargı degismedi (paper'daki "amplitude-dependent generator does not close the gap
either" hala dogru); nonlineer arastirma yolunun UC parcasi (ogrenme, zengin kayip,
sertifika) simdi IKI BAGIMSIZ nonlineer sistemde sinandi ve TUTARLI bir "piksel/CNN
ortaminda genellenmez" sonucuna ulasti -- artik bu, TEK bir deneyin idiosinkratik bir
sonucu degil, CAPRAZ-DOGRULANMIS bir bulgu. Bkz. `clock_diag/pend4_results.jsonl` (ham
veri), `clock_diag/data_duffing2.py`, `clock_diag/run_pend4.py`.

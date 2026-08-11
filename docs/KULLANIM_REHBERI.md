# GeoPINN Studio v4.0 BETA — Kullanım Rehberi

**Beylikova REE-F-Ba-Th Yatağı · Uygulamalı Jeofizik Suite · GeoUNet Öğrenilmiş Ters Çözüm**

---

## Genel Bakış

GeoPINN Studio, yeraltı cevher modellemesi için gravite, manyetik, CSAMT, IP, SP, radyometri, ısı akışı ve **GeoUNet öğrenilmiş ters çözümü** tek bir arayüzde birleştirir. Yerel GPU veya Colab GPU backend üzerinden çalışır.

---

## Arayüz Yapısı

```
┌─────────────────────────────────────────────────────────────────────┐
│  Üst Şerit: GEOPINN v4.0 BETA · Colab GPU · KST IST HRT · BAĞLANTI │
├──────────────┬─────────────────────────────────┬────────────────────┤
│  SOL PANEL   │    ORTA (Navigasyon / İçerik)   │    SAĞ PANEL       │
│  Katmanlar   │                                 │  Analiz (ANALİ)    │
│  Veri        │  ┌──────────┐  ┌─────────────┐ │  PINN / GeoUNet    │
│  Ters Çözüm  │  │ 3D Model │  │ İstatistik  │ │  Belirsizlik       │
│              │  └──────────┘  └─────────────┘ │  SimPEG            │
│  [Analizi    │  ┌──────────┐  ┌─────────────┐ │  FVM               │
│   Başlat]    │  │  Kesit   │  │ Anomali Hrt │ │  Radyometri        │
│  [GeoUNet    │  └──────────┘  └─────────────┘ │  Elektrik (IP/SP)  │
│   Inference] │         2×2 Navigasyon Grid    │  Füzyon            │
│              │  3D butonu → popup modal açar  │  Sismik            │
│              │  KST/HRT/İST → inline panel    │  Geometri          │
│              │                                 │  Filtre            │
│              │                                 │  Dışa Aktar        │
└──────────────┴─────────────────────────────────┴────────────────────┘
```

---

## Adım Adım Çalışma Akışı

### 1. Bağlantı Kurma

Uygulamayı açtığınızda sol üst köşede bağlantı durumu görünür.

- **Colab GPU backend:** Sağ üstteki ⚙ (Ayarlar) ikonuna tıklayın → Colab modunu seçin → ngrok URL'sini girin (örn. `https://xxxx.ngrok-free.app`) → Kaydet
- URL bir kez girilince tarayıcı kapatılıp açılsa bile hatırlanır
- Bağlantı başarılıysa log konsolunda `Backend v3.0 bağlandı` mesajı görünür

---

### 2. Veri Yükleme (Sol Panel → Veri Sekmesi)

#### Desteklenen Formatlar

| Format | Uzantı | İçerik |
|--------|--------|--------|
| GeoPINN model grid | `.npy` | 3D numpy array — geometri veya gözlem |
| Radyometri saha verisi | `.csv`, `.xyz`, `.dat` | x, y, U_ppm, Th_ppm, K_pct |
| SP saha verisi | `.csv`, `.dat` | x, y, SP_mV |
| Gravite saha verisi | `.csv`, `.dat` | x, y, gz_mGal |
| Manyetik saha verisi | `.csv`, `.dat` | x, y, TMI_nT |
| IP verisi | `.csv`, `.dat` | x_mid, depth, chargeability_ms, resistivity_ohmm |
| Sismik refraksiyon | `.csv`, `.dat` | offset_m, tt_ms |

#### Yükleme

1. **Veri formatı** açılır listesinden formatı seçin (veya "Otomatik tespit" bırakın)
2. **Veri Yükle** butonuna tıklayın
3. Sistem formatı otomatik algılar ve beklenen sütunları gösterir

#### Dosya Adlandırma Kuralı (`.npy` dosyaları)

| Önek | Anlam |
|------|-------|
| `Y_...` | Geometri modeli (yeraltı yapısı, cevher gövdesi) |
| `X_mag_grav_...` | Gerçek gravite + manyetik gözlem |
| `X_csamt_...` | Gerçek CSAMT gözlem |

---

### 3. Jeofizik Katman Seçimi (Sol Panel → Katmanlar Sekmesi)

Aktif yöntemleri checkbox ile seçin:

- **Gravite** — Bouguer anomali, yoğunluk kontrastı
- **Manyetik** — TMI (Total Magnetic Intensity), süseptibilite
- **CSAMT** — Görünür özdirenç, derin direnç profili

**Hesaplama Motoru:**
- `Analitik (Prizma)` — Nagy/Bhattacharyya kapalı form, GPU hızlandırmalı, hızlı
- `FVM (Poisson)` — Sınırlı domain Poisson çözücü, daha gerçekçi sınır koşulları

**Analizi Başlat** butonuna basın → backend forward hesaplamasını yapar → 3D model görünür.

---

### 4. 3D Görünüm (Orta Panel)

#### Görünüm Modları (Merkez 2×2 Grid)

| Kart | Açıklama |
|------|----------|
| **3D Model** | Butona tıkla → popup modal açılır (82vw×82vh) · fare ile döndür/zoom |
| **İstatistik** | Değer dağılımı histogramı, P10/P50/P90, korelasyon |
| **Kesit** | X/Y/Z ekseninde 2D dilim, scroll=zoom, sürükle=pan, çift tık=sıfırla |
| **Anomali Haritası** | OSM üzerinde Leaflet overlay, Beylikova koordinatları |

> **Panel boyutu:** Sol ve sağ paneller kenardan tutup sürükleyerek yeniden boyutlandırılabilir (sol: 200–400px, sağ: 280–560px).

#### 3D Kontroller (Sol Panel → Katmanlar → 3D Görünüm; veya popup modal içinde)

- **İzoyüzey eşiği** — düşürünce daha fazla hacim görünür, yükseltince sadece yüksek değerli merkez
- **Şeffaflık** — cevher gövdesinin saydamlığı
- **Arka plan** — Koyu / Açık

---

### 5. Joint Inversion — Ortak Ters Çözüm (Sol Panel → Ters Çözüm Sekmesi)

Birden fazla jeofizik yöntemi aynı yeraltı modeli için aynı anda ters çözümü yapar.

**Adımlar:**
1. Katmanlar sekmesinde en az bir yöntemi aktif edin
2. Ters Çözüm sekmesine geçin
3. Yöntem ağırlıklarını ayarlayın (gravite/manyetik/CSAMT için 0–2 arası)
4. İterasyon sayısını belirleyin (16–150; daha fazla = daha iyi ama yavaş)
5. Grid çözünürlüğünü seçin: `16³` hızlı test, `32³` standart, `64³` native
6. **Ters Çözümü Başlat**

**Sonuç:**
- Yakınsama grafiği (misfit vs iterasyon)
- Yöntemler arası Pearson korelasyonu
- RMSE (gerçek model vs ters çözüm)
- Güncellenmiş 3D model

---

### 6. Belirsizlik Analizi (Sağ Panel → Belirsizlik)

Monte Carlo tabanlı jeolojik belirsizlik ölçümü.

- **Realizasyon sayısı** (3–20): her biri bağımsız gürültülü başlangıçtan ters çözüm
- **Veri gürültüsü (σ)** — gerçek ölçüm hatasını simüle eder (0–%30)
- **Sonuç:** piksel bazında ortalama, standart sapma, P10/P50/P90, Değişim Katsayısı (CV)
- **Yüksek güven bölgesi:** CV < 0.3 — her realizasyonda tutarlı sonuç veren alanlar

---

### 7. SimPEG Tikhonov Inversiyonu (Sağ Panel → SimPEG)

Adjoint tabanlı, gerçek sensitivite matrisli L2 Tikhonov inversiyonu.

**Parametreler:**
- **Smallness (α_s)** — modelin referanstan sapma baskısı; büyük değer = kompakt cevher
- **Smoothness (α_x)** — uzamsal düzgünlük; büyük değer = yumuşak geçişler
- **Chi-faktör** — hedef veri uyuşmazlığı; 1.0 = teorik gürültü seviyesi

**Fark:** Joint Inversion gradient/Adam kullanır (hızlı, yaklaşık). SimPEG Gauss-Newton iterasyonu yapar (yavaş ama analitik sensitivite).

---

### 8. FVM Karşılaştırma (Sağ Panel → FVM)

Analitik prizma motoru ile Sonlu Hacimler Poisson çözücüsünü aynı model üzerinde karşılaştırır.

- **Analitik (Prizma):** Nagy/Bhattacharyya — sonsuz homojen uzay Green fonksiyonu
- **FVM (Poisson):** ∇²U = kaynak — sınırlı domain, Dirichlet sınır koşulları

**Ne zaman FVM kullanılır?**
- Domain sınırına yakın cevher gövdeleri (sınır etkileri önemli)
- Gerçekçi topografya entegrasyonu planlanıyorsa
- Akademik doğrulama gerektiğinde

Karşılaştırma raporu: RMSE, maksimum sapma, göreli hata (%), hesaplama süresi.

---

### 9. Radyometri & Isı Akışı (Sağ Panel → Radyometri)

REE yatağı tespiti için U/Th/K tabanlı analiz.

#### Neden Radyometri?

Beylikova tipi REE-F-Ba-Th yataklarında:
- **Monazit ve xenotim** → Th ve U'yu konsantre eder → yüksek Th/U oranı
- **K-feldispat alterasyonu** → K anomalisi
- **Radyojenik ısı** → U/Th bozunumu → ısı akışı anomalisi → hidrotermal sistemin izi

#### Petrofizik Parametreler

| Parametre | Arka Plan | Cevher Bölgesi |
|-----------|-----------|----------------|
| Uranyum (U) | 3 ppm | 15 ppm |
| Toryum (Th) | 12 ppm | 60 ppm |
| Potasyum (K) | 2.5% | 4.5% |
| Isıl iletkenlik (k) | 2.5 W/m·K | — |

Slider'lar ile Beylikova'ya özgü değerleri ayarlayın.

#### Çıktılar

**REE Hedef İndeksi:**
- `Th/U oranı > 4` → gelişmiş alterasyon, REE mobilizasyonu
- `Bileşik skor > 0.6` → yüksek öncelikli hedef

**Gammaray (yüzey):**
- TC (Total Count) [cps] — yüksek = U+Th+K zenginleşmesi
- Th/U maksimum — alterasyon şiddeti
- Doz hızı [nGy/h] — radyasyon riski referansı

**Radyojenik Isı Akışı:**
- Yüzey ısı akışı [mW/m²]
  - < 40 mW/m² → kraton, soğuk litosfer
  - 60–90 mW/m² → radyojenik zengin granit
  - > 90 mW/m² → aktif hidrotermal sistem, REE yatağı olası
- Isı üretimi Q [μW/m³] — Birch (1954) katsayıları

---

### 10. Referans Geometri Üretici (Sağ Panel → Geometri)

Saha verisi olmadan test/eğitim amaçlı sentetik geometri oluşturur.

| Tip | Açıklama | Beylikova Uyumu |
|-----|----------|-----------------|
| **Hidrotermal Damar** | KB-GD eğimli damar + breş + alterasyon halo | ★★★ En uygun |
| **Borumsu / Pipe** | Düşey eksenli konik yapı | Kimberlite, porfiri Cu |
| **Mercek / Lens** | Yatay elipsoidal mercek | SEDEX, VMS, Au-Ag |
| **Katmana Bağlı** | Yatay stratabound | Cu-Co, PGE, Zn-Pb |

**Parametreler:**
- Eğim açısı (dip): damarın yataydan sapması
- Üst/alt derinlik: cevher gövdesinin derinlik aralığı
- Genişlik/yarıçap
- Breş zonu ve alterasyon halo (açık/kapalı)

Üretilen geometri otomatik olarak `Y_` prefix'li dosya olarak kaydedilir ve dataset listesine eklenir.

---

### 11. Filtre (Sağ Panel → Filtre)

3D model verisine eşikleme ve yumuşatma uygular.

- **Min/Max eşik** — belirli değer aralığındaki vokselleri izole et
- **Gaussian yumuşatma** — gürültüyü azalt, sürekliliği artır
- Filtrelenmiş model 3D görünümde ve kesitte kullanılır

---

### 12. Dışa Aktarma (Sağ Panel → Dışa Aktar)

| Format | İçerik |
|--------|--------|
| CSV | 2D yüzey anomali (x, y, değer) |
| PNG | Mevcut görünümün ekran görüntüsü |
| JSON | Tüm analiz sonuçları ve parametreler |

---

## Önerilen Çalışma Sırası (Beylikova REE Arama)

```
① Veri Yükle / Geometri Üret
   └─ Sol → Veri → .npy yükle  VEYA  Sağ → Geometri → "Hidrotermal Damar"
        ↓
② Analizi Başlat (Forward Modelleme)
   └─ Gravite + Manyetik aktif → Anomali Haritası veya İstatistik kartına tıkla
        ↓
③ GeoUNet Inference  ◄─── YENİ v4.0
   └─ Sol → GeoUNet Inference butonu  VEYA  Sağ → PINN sekmesi
   └─ Kaynak: Y→Forward (sentetik) veya X Dosyası (gerçek anomali)
   └─ Threshold: 0.35–0.45 önerilen · ~0.03s GPU · 32³ çıkış
   └─ "Binary Mask'ı Göster" → eşik üstü vokseller 3D popup'ta
        ↓
④ IP / SP / Radyometri
   └─ Sağ → Elektrik → Cole-Cole + SP mekanizmaları
   └─ Sağ → Radyometri → Th/U haritası + REE hedef skoru
        ↓
⑤ Joint Inversion
   └─ Sol → Ters Çözüm → Grav+Mag+CSAMT → 32³ grid → 60 iter
        ↓
⑥ Füzyon + Belirsizlik
   └─ Sağ → Füzyon → Fuzzy Gamma 7 yöntem
   └─ Sağ → Belirsizlik → CV < 0.3 = güvenilir bölge
        ↓
⑦ 3D Görüntüle (Popup Modal)
   └─ Merkez → "3D Model" kartı → modal açılır
   └─ İzoyüzey eşiği + şeffaflık + koyu/açık arka plan
        ↓
⑧ Dışa Aktar
   └─ Sağ → Dışa Aktar → CSV + PNG → sondaj koordinatları
```

---

## Hızlı Başlangıç (5 dakikada)

1. Uygulamayı aç → **BAĞLANTI** → Colab URL gir → Kaydet
2. Sol panel → **Analizi Başlat** (dataset seçmeden = sentetik Beylikova modeli)
3. Merkez → **3D Model** kartı → popup açılır → izoyüzey incele
4. Sol panel → **GeoUNet Inference** → PINN sekmesi açılır → Çalıştır
5. Sağ panel → **Radyometri** → **Radyometri Hesapla**
6. REE hedef skoru > 0.6 ise → Belirsizlik analizi ile doğrula

---

## GeoUNet — Öğrenilmiş Ters Çözüm (v4.0 YENİ)

### Ne Yapar?

Gravite ve manyetik anomali haritalarından (~0.03 saniyede) 3D cevher geometrisini tahmin eder. Klasik ters çözüm (Adam, SimPEG) yüzlerce iterasyon gerektirirken GeoUNet tek geçişte sonuç üretir.

### Kullanım

1. **Sol panel → GeoUNet Inference** butonu (veya **Sağ → PINN** sekmesi)
2. **Gözlem Kaynağı** seç:
   - `Y → Forward` — seçili Y geometri dosyasından gz+mag hesaplanır (sentetik mod)
   - `X Dosyası` — yüklenen X_mag_grav.npy gerçek anomali haritası kullanılır
3. **Eşik (Threshold)** ayarla:
   - `0.30–0.40` → hassas, geniş cevher bölgesi
   - `0.45–0.55` → dengeli (önerilen başlangıç)
   - `0.60–0.75` → konservatif, yüksek güven çekirdeği
4. **GeoUNet Çalıştır** → sonuçlar sağ panelde görünür
5. **Binary Mask'ı Göster** → eşik üstü vokseller 3D popup'ta

### Sonuç Metrikleri

| Metrik | Açıklama |
|--------|----------|
| Cevher Vokseli | Threshold üstünde kalan 32³ voksel sayısı |
| Cevher Hacmi | Voksel sayısı × (480/32)³ m³ |
| Cevher Fraksiyonu | Toplam hacmin yüzdesi |
| Sigmoid Çıkış | Model güven dağılımı [min, max] ort |
| Cihaz | CUDA (GPU) veya CPU |

### Sınırlamalar

- Model **damar tipi** (vein) geometrilerinde eğitildi — pipe veya lens geometrileri daha az doğru olabilir
- Giriş normalizasyonu eğitim pipeline'ıyla birebir: `gz/5.97×10⁻⁷`, `mag/6.73×10⁻²`
- v4.1 hedefi: 64³ grid, CSAMT kanalı, daha yüksek IoU


---

## Teknik Notlar

**Koordinat sistemi:** UTM Zone 36N (Beylikova), domain 480 × 480 × 480 m

**Petrofizik bağıntılar:**
- Yoğunluk: ρ(x) = 2.70 + 2.00 × f(x) g/cm³
- Süseptibilite: χ(x) = 1×10⁻⁴ + 3×10⁻⁴ × f(x) SI
- Özdirenç: ρₑ(x) = 500 × 0.10^f(x) Ω·m

**GPU Kullanımı:** Gravite, manyetik, CSAMT, joint inversion ve GeoUNet PyTorch/CUDA üzerinde çalışır. SimPEG ve FVM CPU tabanlıdır.

**Grid boyutları:**
- `16³` — hızlı test (saniyeler)
- `32³` — standart analiz (dakikalar)
- `64³` — native çözünürlük (uzun süre, yüksek bellek)

## [v3.1.1] — 2026-08-08

### Düzeltmeler

- `jiHistory?.length` — null crash fix (istatistik siyah ekran)
- SP ölçek normalizasyonu — -2628 mV → gerçekçi ±500 mV aralığı
- `GRAV_AVAILABLE` / `MAG_AVAILABLE` flag'leri eklendi — füzyon Internal Server Error
- `heat_flow_fvm.py` import düzeltmesi — `from fvm_core` → try/except with `engines.fvm_core`
- Split view sistemi — floating pencere yerine 3D viewer yanında panel (İST/KST/HRT)
- Sağ panel genişliği 256 → 320px
- Log konsolu 180 → 110px
- Rehber güncellendi: IP & SP, Füzyon sekmeleri eklendi
- İş akışı 6 → 8 adım: IP/SP (③), Radyometri (④), Füzyon (⑤)

### Notlar

IP ve SP forward motorları şu an **sadece forward model** — ters çözüme (joint inversion)
dahil değiller. IP/SP çıktıları füzyon aracılığıyla sondaj hedef seçimini destekler.
IP/SP inversion entegrasyonu v4.0 PINN çerçevesiyle birlikte planlanmaktadır.

---

## [v3.1.0] — 2026-08-07

### Yeni Fizik Motorları

**IP (Induced Polarization) — `engines/ip_forward.py`**
- Cole-Cole kompleks özdirenç modeli — Pelton et al. (1978), *Geophysics* 43(3)
- Dipole-dipole pseudosection — Telford et al. (1990)
- Pseudo-derinlik z≈0.519·n·a — Loke & Barker (1996)
- Görünür chargeability — Seigel (1959)
- Beylikova: pirit/arsenopirit → m>0.3, ρ₀<20 Ω·m → güçlü IP hedefi
- `/api/ip/forward` + `/api/ip/status`

**SP (Self-Potential) — `engines/sp_forward.py`**
- Elektrokinetik kuplaj: L_ek = -ε₀εᵣζσ/(ηF) — Revil & Leroy (2004), *JGR*
- Termoelektrik kuplaj (Seebeck): L_T = T_s·σ — Revil et al. (2012)
- Elektrokimyasal battery model — Sato & Mooney (1960), Mendonça (2008)
- Green fonksiyonu yüzey projeksiyonu — Sill (1983)
- Beylikova beklenen: -50 ile -300 mV (hidrotermal + sülfür)
- `/api/sp/forward` + `/api/sp/status`

**Data Fusion — `engines/data_fusion.py`**
- 7 yöntem bileşik anomali skoru: Gravite, Manyetik, CSAMT, IP, SP, Radyometri, Isı Akışı
- Füzyon yöntemleri: Fuzzy Gamma (Zimmermann & Zysno 1980), Ağırlıklı Toplam,
  Fuzzy AND/OR, Index Overlay (Carranza & Hale 2002)
- Otomatik hedef bölge tespiti (scipy.ndimage.label)
- Yöntem korelasyon matrisi
- Referans: Porwal et al. (2003), Bonham-Carter (1994)
- `/api/fusion/composite` + `/api/fusion/status`

### Frontend Değişiklikleri

**Elektrik Sekmesi (Sağ Panel)**
- IP ve SP toggle menüsünde birleştirildi
- Cole-Cole parametreleri slider ile (host/cevher ayrı)
- SP mekanizma ağırlıkları (EK/TE/EC)
- Sonuç kartları: ρₐ, chargeability, faz, SP anomali, baskın kaynak

**Füzyon Sekmesi (Sağ Panel)**
- 7 yöntemden istenen seçilir, checkbox ile
- Füzyon yöntemi dropdown (5 seçenek)
- Öncelikli hedef bölgeler listesi (merkez koordinat, skor, piksel sayısı)
- Yöntem korelasyon matrisi (renk kodlu: yeşil=yüksek, sarı=orta)

**Katmanlar Sekmesi**
- IP ve SP katman seçenekleri eklendi (Grav/Mag/CSAMT yanına)

---

## [v3.0.0.2] — 2026-08-06

### Yeni Özellikler

**Leaflet.js Anomali Haritası**
- OSM tile layer üzerine model anomalisi `ImageOverlay` olarak yükleniyor
- Gerçek koordinat sistemi — Beylikova (39.92°N, 31.67°E), 480×480m alan
- Çalışma alanı sınırı turuncu kesik çizgi, merkez marker
- Mavi→yeşil→sarı→kırmızı jeofizik renk skalası
- Dinamik CDN yükleme — `package.json` değişikliği gerekmez

**Ayrı Pencere Sistemi (Electron)**
- KST/İST/HRT butonlarından gerçek `BrowserWindow` açılıyor
- Uygulama sınırları dışına çıkabiliyor, monitörler arası taşınabiliyor
- `localStorage` paylaşımı — analiz verisi otomatik floating panele yansıyor
- Panel penceresi kapanınca ana uygulama etkilenmiyor

**Custom Titlebar & Logo**
- `frame: false` — native Windows başlık çubuğu kaldırıldı
- Prizma katman logo (C konsepti) — GP monogram, turuncu katmanlar, derinlik çizgileri
- Özel ─ ▢ ✕ butonları (`WebkitAppRegion: no-drag`)
- `Menu.setApplicationMenu(null)` — boş File/Edit/View menü barı kaldırıldı

**Backend CUDA Otomatik Tespit**
- `torch.cuda.is_available()` ile yerel NVIDIA GPU otomatik kullanılıyor
- Colab bağlantısı opsiyonel — lokal GPU varsa gerek yok
- `[BİLGİ] HESAPLAMA DONANIMI: CUDA` logu ile doğrulanıyor

**SimPEG Split Section**
- Gravite ve manyetik sonuçları ayrı kartlarda — her biri kendi Gauss-Newton grafiğiyle
- Adam vs SimPEG fark açıklaması
- Koordinat sistemi düzeltmesi — mesh origin ve obs_pts Z koordinatı

### Düzeltmeler

- `setSimPEGAvailable` → `setSimpegAvailable` typo düzeltmesi (siyah ekran)
- Panel penceresi `window-close` IPC — `event.sender` ile hangi pencere belirleniyor
- `WebkitAppRegion: drag` içinde buton tıklanamama sorunu
- `viewTabs` sadece 3D — Kesit/Harita/İstatistik ayrı pencere olarak açılıyor
- Backend bağlantı sıralaması — modül status'ları `backendOk=true` sonrası kontrol ediliyor
- Leaflet popup içinde tek tırnak syntax hatası

---

# CHANGELOG — GeoPINN Studio

---

## [v3.0.0.1] — 2026-07-30

### Yeni Özellikler

**Radyometri & Isı Akışı Motoru**
- `engines/radiometry.py` — U/Th/K konsantrasyonlarından gammaray sayımı forward modeli
- `engines/heat_flow_fvm.py` — FVM tabanlı radyojenik ısı iletimi (∇²T = -Q/k)
- Th/U oranı, eU (efektif uranyum), bileşik REE anomali skoru
- Beylikova REE-F-Ba-Th yatağı için özelleştirilmiş petrofizik parametreler
- `/api/radiometry/forward`, `/api/radiometry/status`, `/api/radiometry/upload-field-data` endpoint'leri

**FVM (Sonlu Hacimler) Motor Entegrasyonu**
- `engines/gravity_fvm.py` + `engines/magnetic_fvm.py` — Poisson bounded domain çözücü
- Analitik prizma motoru (Nagy/Bhattacharyya) ile yan yana karşılaştırma
- `engine_mode` parametresi: `prism` veya `fvm` seçimi
- `/api/fvm/status`, `/api/fvm/compare` endpoint'leri
- RMSE, maksimum sapma, göreli hata (%), hesaplama süresi raporu

**Veri Formatı Seçici**
- 7 farklı jeofizik veri formatı desteği: radyometri CSV, SP, IP, sismik refraksiyon, GPR, gravite, manyetik
- Otomatik format tespiti (başlık satırı analizi)
- `/api/data/formats`, `/api/data/detect-format` endpoint'leri

**Arayüz Yenileme**
- Çift tema sistemi: tam karanlık (`#0A0C0F`) ↔ tam aydınlık, localStorage'da kalıcı
- Sağ panel: yatay sekme çubuğu → dikey ikon rail (8 sekme sığdı, taşma yok)
- Üst şerit: sismik profil imza elementi, GEO/PINN renk ayrımı
- Help modal: GitHub repo linki + iletişim e-postası, Radyometri ve FVM sekmeleri
- Pastel renk paleti kaldırıldı, keskin kenarlıklar (`1.5px solid`)

**Electron (Masaüstü)**
- Uygulama açılışında tam ekran (maximize)
- Pencere kapatınca sistem tepsisine küçülme (Windows/Linux)
- Tepsi menüsü: Göster, Tam Ekran, Çıkış
- `main.cjs` pencere kontrol IPC handler'ları

**Yeni Endpoint'ler**
- `/api/seismic/vp-classes` — Q1-Q6 kaya sınıfı tablosu
- `/api/methods` — aktif motor listesi ve müsaitlik durumu
- `/api/radiometry/*` — radyometri suite
- `/api/fvm/*` — FVM karşılaştırma
- `/api/data/formats`, `/api/data/detect-format`

### İyileştirmeler

- Joint Inversion: SPSA → Adam + autograd (gerçek gradyan, daha hızlı yakınsama)
- SimPEG, FVM, Radyometri modülleri yüklü değilse `null` göster (`false` değil) — bağlantı kurulmadan hata mesajı çıkmıyor
- `useEffect` dependency array'lerine `apiBase` eklendi — Colab URL değişince motorlar otomatik yeniden kontrol ediliyor
- `n_iter` slider max 200 → 150 (backend limiti ile eşleşti)
- Duplicate `workflow` ve `VP_CLASS_COLORS` tanımları temizlendi

### Düzeltmeler

- `/api/uncertainty` HTTP 404 — patch ile server.py'e eklendi
- SimPEG "Kurulu Değil" yanlış gösterimi — apiBase bağımlılığı düzeltildi
- JSX aside/div tag eşleşme hataları — build hatası giderildi
- `theme is not defined` ReferenceError — `useTheme()` App fonksiyonu içine alındı
- ngrok 5 tunnel limiti — `ngrok.kill()` ile temizleme adımı eklendi

### Known Issues / Dikkat

- **reg_lambda hassasiyeti:** `0.05` üzerinde düzenlileştirme misfit'i bozabilir. Şu an manuel ayar gerekiyor — v3.1'de GradNorm adaptive balancing planlanıyor.
- **Yüksek grid maliyeti:** `32³` üzeri FVM ve SimPEG inversiyonları CPU-bound, bellek yoğun.
- **Jeolojik validasyon eksik:** Kütle/hacim tahminleri model-tutarlı, sondaj verisiyle çapraz doğrulanmamış.
- **1D CSAMT kısıtı:** Karmaşık 3D yapılarda 2D/3D etkiler modellere yansımıyor.

---

## [v3.0.0] — 2026-07-01

### Temel Sürüm

**Fizik Motorları**
- `gravity_prism.py` — Nagy analitik gravite (GPU, PyTorch)
- `magnetic_prism.py` — Bhattacharyya TMI manyetik (GPU, PyTorch)
- `csamt_1d.py` — Wait özyineleme MT/CSAMT (GPU, PyTorch)
- `fvm_core.py` — Poisson FVM çekirdek (scipy.sparse)
- `petrophysics.py` — Beylikova petrofizik bağıntıları
- `harmonica_validation.py` — Fatiando a Terra doğrulama

**Ters Çözüm**
- Joint Inversion: Adam optimizer + autograd, normalize MSE loss
- Belirsizlik Analizi: Monte Carlo realizasyonları, CV/P10/P90
- SimPEG Tikhonov L2: adjoint tabanlı Gauss-Newton

**Veri Yönetimi**
- `.npy` upload/download, shape ve tip sınıflandırması
- Analiz kayıt/listeleme/silme (`/api/analyses`)
- Referans geometri üretici (damar, pipe, lens, stratabound)

**Frontend**
- Three.js isosurface (MarchingCubes) 3D görünüm
- Kesit görünümü (X/Y/Z), harita, istatistik sekmeleri
- Recharts yakınsama grafiği
- Colab GPU bağlantı yönetimi (ngrok URL, localStorage kalıcı)
- Electron masaüstü paketi (Windows ZIP)

---

## Yol Haritası

### v3.1.0 (Planlanan)
- `data_fusion.py` — tüm metodların ortak grid bileşik anomali skoru (Co-kriging)
- IP (Induced Polarization) forward motoru — Cole-Cole kompleks özdirenç
- SP (Öz-potansiyel) forward motoru — elektrokinetik kuplaj (∇²V = ∇·(L·∇T))
- Gerçek saha verisi grid'leme (düzensiz → düzenli, RBF interpolasyon)

### v4.0.0 (PINN Entegrasyonu)
- Physics-Informed Neural Networks — Laplace/Poisson PDE kayıp fonksiyonu
- PyTorch autograd uyumlu fizik katmanı (`physics_eqs.py`)
- Çok-yöntemli PINN joint inversion
- Transfer learning: sentetik → gerçek saha verisi

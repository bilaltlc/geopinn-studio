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

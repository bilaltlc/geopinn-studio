"""
geometry/gaussian_bodies.py

Beylikova (Kizilcaoren) tipi REE-F-Ba-Th yatağının jeolojik morfolojisini
temsil eden anizotropik Gaussian gövde üreticisi.

JEOLOJIK GERCEKLIK (kaynak: mindat.org / literatur):
    "Mineralizasyon, damar dolgulari (vein-fillings), piroklastik kayalari
     cimentolayan bres cevherleri (breccia ores) ve piroklastik kayalar
     icinde 50 m'ye kadar kalinlikta mercek seklinde tabakali cevher
     kutleleri (lens-shaped stratified ore bodies) seklinde olusur."

Bu nedenle yer altindaki yogunluk/manyetik-duyarlilik/ozdirenc kontrast
alani UC ayri geometrik bilesenin lineer toplami olarak insa edilir:

    1. LENS   : yatayda genis, dusayda (z) dar, yatay duzlemde rastgele
                dondurulmus anizotropik Gaussian. "50 m'ye kadar kalinlik"
                kisitini sigma_z araligiyla dogrudan kodluyoruz.
    2. VEIN   : tek eksende cok uzun, diger iki eksende dar, rastgele
                egim (dip) acisiyla 3D'de dondurulmus anizotropik Gaussian.
    3. BRECCIA: kucuk izotropik Gaussian'larin bir ana merkez etrafinda
                rastgele dagilmis kumesi (Neyman-Scott tipi nokta sureci
                mantigi - "kume merkezi" + "kume icindeki noktalar").

TEORI - Anizotropik 3D Gaussian:
    f(x) = A * exp( -0.5 * (x-c)^T Sigma^-1 (x-c) )

    Sigma (kovaryans matrisi) izotropik oldugunda (sigma_x=sigma_y=sigma_z,
    eksenler arasi korelasyon yok) sekil bir KUREDIR. Biz Sigma'yi
    sekillendirerek (farkli sigma'lar + rotasyon matrisi R) mercek,
    damar gibi anizotropik formlar uretiyoruz:

        x' = R^T (x - c)          # global koordinati govde-lokal eksene tasi
        f(x) = A * exp( -0.5 * sum_i (x'_i / sigma_i)^2 )   # eksene-hizali anizotropik Gaussian

    Bu, "dondurulmus elipsoid Gaussian" elde etmenin standart yontemidir.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 3D Rotasyon matrisi yardimcisi
# ---------------------------------------------------------------------------

def rotation_matrix_zyx(strike_deg: float, dip_deg: float, rake_deg: float = 0.0) -> np.ndarray:
    """
    Jeofizikte yaygin kullanilan strike/dip/rake aciları ile 3D rotasyon matrisi.

    strike_deg : yatay duzlemde donus acisi (z ekseni etrafinda), derece
    dip_deg    : dusey duzlemde egim acisi (y ekseni etrafinda), derece
    rake_deg   : govde-lokal eksen etrafinda ek donus (x ekseni etrafinda), derece

    Donus sirasi: once rake (x), sonra dip (y), sonra strike (z).
    R = Rz(strike) @ Ry(dip) @ Rx(rake)
    """
    s, d, r = np.radians([strike_deg, dip_deg, rake_deg])

    Rz = np.array([
        [np.cos(s), -np.sin(s), 0],
        [np.sin(s),  np.cos(s), 0],
        [0,          0,         1],
    ])
    Ry = np.array([
        [np.cos(d),  0, np.sin(d)],
        [0,          1, 0],
        [-np.sin(d), 0, np.cos(d)],
    ])
    Rx = np.array([
        [1, 0,          0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)],
    ])
    return Rz @ Ry @ Rx


def anisotropic_gaussian(X, Y, Z, center, sigmas, R, amplitude=1.0):
    """
    Dondurulmus, eksene-hizalanmamis (rotated) anizotropik 3D Gaussian.

    X, Y, Z   : meshgrid koordinat dizileri (ayni sekil)
    center    : (cx, cy, cz) govde merkezi
    sigmas    : (sx, sy, sz) govde-lokal eksenlerdeki yayilim genislikleri
    R         : 3x3 rotasyon matrisi (govde-lokal -> global donusum)
    amplitude : tepe genligi
    """
    cx, cy, cz = center
    sx, sy, sz = sigmas

    # Global koordinattan govde-lokal koordinata gec: x' = R^T (x - c)
    dx = X - cx
    dy = Y - cy
    dz = Z - cz

    # R^T uygulanmasi: lokal_i = sum_j R[j,i] * delta_j  (R^T @ delta)
    xl = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
    yl = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
    zl = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz

    exponent = -0.5 * ((xl / sx) ** 2 + (yl / sy) ** 2 + (zl / sz) ** 2)
    return amplitude * np.exp(exponent)


# ---------------------------------------------------------------------------
# Govde parametre kayitlari (dataclass) - okunabilirlik ve test edilebilirlik icin
# ---------------------------------------------------------------------------

@dataclass
class LensBody:
    """Mercek sekilli tabakali cevher kutlesi (lens-shaped stratified ore body)."""
    center: tuple[float, float, float]
    sigma_xy: float          # yatay yayilim (m), genis
    sigma_z: float           # dusey yayilim (m), dar -> "50 m'ye kadar kalinlik"
    strike_deg: float        # yatay donus acisi
    amplitude: float = 1.0

    def field(self, X, Y, Z):
        R = rotation_matrix_zyx(self.strike_deg, dip_deg=0.0)
        return anisotropic_gaussian(
            X, Y, Z, self.center,
            sigmas=(self.sigma_xy, self.sigma_xy, self.sigma_z),
            R=R, amplitude=self.amplitude,
        )


@dataclass
class VeinBody:
    """Ince, uzamis damar dolgusu (vein-filling)."""
    center: tuple[float, float, float]
    length_sigma: float      # uzun eksen yayilimi (m)
    width_sigma: float       # dar eksenler yayilimi (m)
    strike_deg: float
    dip_deg: float
    amplitude: float = 1.0

    def field(self, X, Y, Z):
        R = rotation_matrix_zyx(self.strike_deg, self.dip_deg)
        return anisotropic_gaussian(
            X, Y, Z, self.center,
            sigmas=(self.length_sigma, self.width_sigma, self.width_sigma),
            R=R, amplitude=self.amplitude,
        )


@dataclass
class BrecciaCluster:
    """
    Bres cevher kumesi: bir ana merkez etrafinda dagilmis N kucuk izotropik
    Gaussian blob. Neyman-Scott nokta sureci mantigi: "ebeveyn" nokta (parent)
    + ebeveyn etrafinda Gaussian dagilmis "cocuk" noktalar (children/offspring).
    """
    parent_center: tuple[float, float, float]
    n_blobs: int
    spread: float             # cocuk noktalarin ebeveyn etrafindaki dagilim genisligi (m)
    blob_sigma_range: tuple[float, float]   # her blobun kendi sigma araligi (m)
    amplitude_range: tuple[float, float]
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def field(self, X, Y, Z):
        total = np.zeros_like(X, dtype=np.float32)
        px, py, pz = self.parent_center
        for _ in range(self.n_blobs):
            # Cocuk merkezi: ebeveyn etrafinda izotropik Gaussian sacilma
            offset = self.rng.normal(scale=self.spread, size=3)
            blob_center = (px + offset[0], py + offset[1], pz + offset[2])

            sigma = self.rng.uniform(*self.blob_sigma_range)
            amp = self.rng.uniform(*self.amplitude_range)

            R = np.eye(3)  # izotropik blob icin rotasyon onemsiz
            total += anisotropic_gaussian(
                X, Y, Z, blob_center,
                sigmas=(sigma, sigma, sigma),
                R=R, amplitude=amp,
            )
        return total


# ---------------------------------------------------------------------------
# Sentetik Beylikova-tipi model uretici
# ---------------------------------------------------------------------------

@dataclass
class BeylikovaModelConfig:
    """
    Beylikova/Kizilcaoren tipi tek bir sentetik model orneginin uretim
    parametreleri. Degerler, "50 m'ye kadar mercek kalinligi" jeolojik
    kisitiyla ve tipik orta-olcekli REE-karbonatit govde boyutlariyla
    tutarli secilmistir (PoC asamasinda gercekci buyukluk mertebesi
    onceliklidir, sahaya-ozgu kesin sinirlar degil).
    """
    grid_extent_m: float = 480.0     # model kutusunun her kenari (m)
    grid_size: int = 80               # FVM hucre sayisi her eksende
    depth_top_m: float = 50.0        # govdelerin baslayabilecegi en sig derinlik
    depth_bottom_m: float = 350.0    # govdelerin bulunabilecegi en derin nokta

    n_lenses: tuple[int, int] = (1, 2)
    n_veins: tuple[int, int] = (1, 3)
    n_breccia_clusters: tuple[int, int] = (1, 2)

    # NOT (grid-cozunurluk uzlasmasi): dh = grid_extent_m/(grid_size-1) = 480/79 = 6.08 m.
    # Asagidaki genislik/sigma araliklari, en ince yapinin (vein_width alt siniri)
    # en az ~1.6 hucreye yayilmasini saglayacak sekilde, gercek jeolojik
    # kalinliktan biraz daha "sisman" secilmistir (Nyquist guvenligi icin
    # bilincli bir PoC uzlasma -- gercek damarlar genelde <1m olabilir,
    # ama grid'de temsil edilemeyen yapi forward modelde gorunmez kalir).
    lens_sigma_xy_range: tuple[float, float] = (60.0, 150.0)
    lens_sigma_z_range: tuple[float, float] = (15.0, 32.0)     # <=50m kalinlik kisitina uyumlu (FWHM ~2.35*sigma)

    vein_length_sigma_range: tuple[float, float] = (90.0, 200.0)
    vein_width_sigma_range: tuple[float, float] = (10.0, 18.0)

    breccia_spread_range: tuple[float, float] = (30.0, 70.0)
    breccia_blob_sigma_range: tuple[float, float] = (12.0, 22.0)
    breccia_n_blobs_range: tuple[int, int] = (3, 8)

    seed: int | None = None


class BeylikovaSyntheticModel:
    """
    Verilen konfigurasyona gore lens + vein + breccia bilesenlerini rastgele
    orneklenmis parametrelerle uretir ve bunlarin toplamini (normalize
    edilmemis "anomali siddet" alani olarak) dondurur.

    Bu alan SONRADAN petrofiziksel kontrast degerleriyle (Delta-yogunluk,
    Delta-duyarlilik, Delta-ozdirenc) olceklenip gercek fiziksel birimlere
    (g/cm3, SI, ohm.m) cevrilir - bkz. petrophysics.py
    """

    def __init__(self, config: BeylikovaModelConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        n = config.grid_size
        ext = config.grid_extent_m
        coords = np.linspace(0, ext, n)
        self.X, self.Y, self.Z = np.meshgrid(coords, coords, coords, indexing="ij")
        self.dh = ext / (n - 1)  # hucre boyutu (m)

        self.bodies: list = []

    def _random_xy(self):
        ext = self.config.grid_extent_m
        margin = ext * 0.15
        return self.rng.uniform(margin, ext - margin), self.rng.uniform(margin, ext - margin)

    def _random_depth(self):
        return self.rng.uniform(self.config.depth_top_m, self.config.depth_bottom_m)

    def generate(self) -> np.ndarray:
        cfg = self.config
        field_total = np.zeros_like(self.X, dtype=np.float32)
        self.bodies = []

        n_lenses = self.rng.integers(cfg.n_lenses[0], cfg.n_lenses[1] + 1)
        for _ in range(n_lenses):
            cx, cy = self._random_xy()
            cz = self._random_depth()
            body = LensBody(
                center=(cx, cy, cz),
                sigma_xy=self.rng.uniform(*cfg.lens_sigma_xy_range),
                sigma_z=self.rng.uniform(*cfg.lens_sigma_z_range),
                strike_deg=self.rng.uniform(0, 360),
                amplitude=1.0,
            )
            field_total += body.field(self.X, self.Y, self.Z)
            self.bodies.append(body)

        n_veins = self.rng.integers(cfg.n_veins[0], cfg.n_veins[1] + 1)
        for _ in range(n_veins):
            cx, cy = self._random_xy()
            cz = self._random_depth()
            body = VeinBody(
                center=(cx, cy, cz),
                length_sigma=self.rng.uniform(*cfg.vein_length_sigma_range),
                width_sigma=self.rng.uniform(*cfg.vein_width_sigma_range),
                strike_deg=self.rng.uniform(0, 360),
                dip_deg=self.rng.uniform(20, 90),  # dik/yari-dik damarlar tipik
                amplitude=1.0,
            )
            field_total += body.field(self.X, self.Y, self.Z)
            self.bodies.append(body)

        n_breccia = self.rng.integers(cfg.n_breccia_clusters[0], cfg.n_breccia_clusters[1] + 1)
        for _ in range(n_breccia):
            cx, cy = self._random_xy()
            cz = self._random_depth()
            body = BrecciaCluster(
                parent_center=(cx, cy, cz),
                n_blobs=int(self.rng.integers(*cfg.breccia_n_blobs_range)),
                spread=self.rng.uniform(*cfg.breccia_spread_range),
                blob_sigma_range=cfg.breccia_blob_sigma_range,
                amplitude_range=(0.5, 1.0),
                rng=self.rng,
            )
            field_total += body.field(self.X, self.Y, self.Z)
            self.bodies.append(body)

        # Anomali siddetini [0, 1] araligina normalize et (petrophysics.py
        # bu normalize alani gercek fiziksel kontrastlarla olcekleyecek).
        max_val = field_total.max()
        if max_val > 1e-12:
            field_total = field_total / max_val

        return field_total
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    
    print("[BAŞLADI] Geometri üretici bağımsız testi çalıştırılıyor...", flush=True)

    # Konfigürasyonu kilitli seed (42) ile çağır ki hep aynı kararlı çıktıyı göresin
    config = BeylikovaModelConfig(seed=42)
    model = BeylikovaSyntheticModel(config)
    
    print("-> Matris hesaplanıyor (Bu islem Ryzen 5 2600 üzerinde birkaç saniye sürebilir)...", flush=True)
    field_3d = model.generate()

    print(f"-> Grid Boyutu: {field_3d.shape}", flush=True)
    print(f"-> Hücre Boyutu (dh): {model.dh:.2f} metre", flush=True)
    print(f"-> Değer Aralığı: Min = {field_3d.min():.4f}, Maks = {field_3d.max():.4f}", flush=True)

    # Maksimum anomali hücresinin (cevher merkezinin) indeksini bul
    max_idx = np.unravel_index(np.argmax(field_3d), field_3d.shape)
    
    print(f"-> Kesitler çiziliyor (Merkez Indeks: X={max_idx[0]}, Y={max_idx[1]}, Z={max_idx[2]})...", flush=True)
    
    # 3'lü subplot oluştur
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. XY Kesiti (Üstten Görünüş)
    im0 = axes[0].imshow(field_3d[:, :, max_idx[2]].T, origin="lower", extent=[0, 480, 0, 480], cmap="inferno")
    axes[0].set_title(f"XY Kesiti (z_idx={max_idx[2]})")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    fig.colorbar(im0, ax=axes[0])

    # 2. XZ Kesiti (Yandan Görünüş)
    im1 = axes[1].imshow(field_3d[:, max_idx[1], :].T, origin="upper", extent=[0, 480, 480, 0], cmap="inferno")
    axes[1].set_title(f"XZ Kesiti (y_idx={max_idx[1]})")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z - Derinlik (m)")
    fig.colorbar(im1, ax=axes[1])

    # 3. YZ Kesiti (Önden Görünüş)
    im2 = axes[2].imshow(field_3d[max_idx[0], :, :].T, origin="upper", extent=[0, 480, 480, 0], cmap="inferno")
    axes[2].set_title(f"YZ Kesiti (x_idx={max_idx[0]})")
    axes[2].set_xlabel("Y (m)")
    axes[2].set_ylabel("Z - Derinlik (m)")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    
    # Görüntüyü projenin kök dizinine kaydet
    output_path = "geometry_check_local.png"
    plt.savefig(output_path, dpi=150)
    print(f"[BAŞARILI] Kesit grafiği '{output_path}' olarak kaydedildi.", flush=True)
    
    # Eğer VS Code'da bir X server veya GUI desteği varsa pencereyi de açar
    plt.show()
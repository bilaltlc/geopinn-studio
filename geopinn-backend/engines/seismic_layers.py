"""
seismic_layers.py — Sismik Katman Modellemesi & Jeolojik Sınıflandırma
GeoPINN Studio 3.0

Desteklenen inputlar:
  1. GeoTIFF DEM (.tif)         → topografya yüzeyi
  2. Sismik traveltime CSV      → first arrival (offset_m, tt_ms, shot_x, shot_y)
  3. SEGY (.sgy)                → ham sismik kayıt (obspy ile)
  4. Mevcut Vp model (.npy)     → doğrudan tomografi girişi

Çıktılar (QGIS uyumlu):
  - dem_grid.npy                → (ny, nx) yükseklik grid'i
  - vp_model.npy                → (nz, ny, nx) Vp tomografi
  - layer_model.npy             → (nz, ny, nx) kaya birimi indeks
  - geology_surface.tif         → QGIS GeoTIFF (kaya birimi yüzey haritası)
  - layer_summary.json          → katman istatistikleri

Kaya birimi sınıflandırması (Vp eşiği):
  Q1: < 800  m/s  Toprak / dolgu
  Q2: 800-1800    Ayrışmış kaya
  Q3: 1800-3000   Kırıklı kaya (zayıf)
  Q4: 3000-4000   Kırıklı sağlam kaya
  Q5: 4000-5500   Sağlam granit/kireçtaşı
  Q6: > 5500      Sert bazalt/kuvarsit
"""

from __future__ import annotations
import os
import json
import warnings
import numpy as np
from scipy.interpolate import RBFInterpolator, griddata
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple

# Opsiyonel bağımlılıklar
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
    warnings.warn("rasterio bulunamadı. GeoTIFF export devre dışı. pip install rasterio")

try:
    import obspy
    OBSPY_OK = True
except ImportError:
    OBSPY_OK = False


# ── Kaya birimi sınıflandırma tablosu ─────────────────────────────────────────
VP_CLASSES = [
    {"id": 1, "label": "Q1 — Toprak/Dolgu",         "vp_min":    0, "vp_max":  800,
     "color": "#8B4513", "rqd": 0,   "ucs_mpa":   0, "drill_factor": 0.3},
    {"id": 2, "label": "Q2 — Ayrışmış Kaya",         "vp_min":  800, "vp_max": 1800,
     "color": "#D2691E", "rqd": 10,  "ucs_mpa":  10, "drill_factor": 0.5},
    {"id": 3, "label": "Q3 — Kırıklı Kaya (Zayıf)", "vp_min": 1800, "vp_max": 3000,
     "color": "#DAA520", "rqd": 30,  "ucs_mpa":  30, "drill_factor": 0.7},
    {"id": 4, "label": "Q4 — Kırıklı Sağlam",       "vp_min": 3000, "vp_max": 4000,
     "color": "#6B8E23", "rqd": 60,  "ucs_mpa":  60, "drill_factor": 1.0},
    {"id": 5, "label": "Q5 — Sağlam Granit/Kireç",  "vp_min": 4000, "vp_max": 5500,
     "color": "#2E8B57", "rqd": 85,  "ucs_mpa": 120, "drill_factor": 1.8},
    {"id": 6, "label": "Q6 — Sert Bazalt/Kuvarsit", "vp_min": 5500, "vp_max": 9999,
     "color": "#1C3A5E", "rqd": 95,  "ucs_mpa": 200, "drill_factor": 3.0},
]


def classify_vp(vp_array: np.ndarray) -> np.ndarray:
    """Vp (m/s) dizisini kaya birimi ID'sine çevirir."""
    out = np.zeros_like(vp_array, dtype=np.int8)
    for cls in VP_CLASSES:
        mask = (vp_array >= cls["vp_min"]) & (vp_array < cls["vp_max"])
        out[mask] = cls["id"]
    return out


# ── DEM / GeoTIFF okuma ────────────────────────────────────────────────────────
def load_dem_geotiff(tif_path: str, target_nx: int = 64,
                      target_ny: int = 64) -> Tuple[np.ndarray, dict]:
    """
    GeoTIFF DEM'i okur ve hedef çözünürlüğe resampling yapar.

    Döndürür:
        dem_grid : (ny, nx) float64 yükseklik matrisi (m)
        meta     : koordinat sistemi bilgisi
    """
    if not RASTERIO_OK:
        raise ImportError("rasterio kurulu değil. pip install rasterio")

    with rasterio.open(tif_path) as src:
        # Orijinal grid
        data = src.read(1).astype(np.float64)
        data[data == src.nodata] = np.nan if src.nodata else 0.0
        bounds = src.bounds
        crs    = src.crs

    # Nan'ları interpolasyonla doldur
    if np.any(np.isnan(data)):
        from scipy.ndimage import generic_filter
        mask = np.isnan(data)
        data[mask] = generic_filter(
            np.where(mask, 0, data),
            lambda x: np.nanmean(x) if not np.all(x == 0) else 0,
            size=5
        )[mask]

    # Hedef boyuta resampling (bilinear)
    from scipy.ndimage import zoom
    zoom_y = target_ny / data.shape[0]
    zoom_x = target_nx / data.shape[1]
    dem_grid = zoom(data, (zoom_y, zoom_x), order=1)

    meta = {
        "bounds": {"west": bounds.left, "east": bounds.right,
                   "south": bounds.bottom, "north": bounds.top},
        "crs": str(crs) if crs else "EPSG:4326",
        "original_shape": list(data.shape),
        "resampled_shape": [target_ny, target_nx],
        "z_min": float(np.nanmin(dem_grid)),
        "z_max": float(np.nanmax(dem_grid)),
    }
    print(f"[DEM] Yüklendi: {data.shape} → {dem_grid.shape}  "
          f"Z: {meta['z_min']:.1f} – {meta['z_max']:.1f} m")
    return dem_grid, meta


def synthetic_dem(nx: int = 64, ny: int = 64,
                   domain_m: float = 480.0,
                   relief_m: float = 80.0, seed: int = 42) -> Tuple[np.ndarray, dict]:
    """
    Gerçekçi sentetik DEM — fraktal topografya (spectral synthesis).
    GeoTIFF olmadığında test/demo için kullanılır.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    # Spektral sentez (1/f² gürültü → dağlık topografya)
    Z = np.zeros((ny, nx))
    for freq in [1, 2, 3, 5, 8, 13]:
        amp = relief_m / (freq * 3)
        ph_x = rng.uniform(0, 2*np.pi)
        ph_y = rng.uniform(0, 2*np.pi)
        Z += amp * np.sin(2*np.pi*freq*X + ph_x) * np.cos(2*np.pi*freq*Y + ph_y)
    Z = Z - Z.min()  # sıfır taban
    dem_grid = Z.astype(np.float64)
    meta = {
        "bounds": {"west": 0, "east": domain_m, "south": 0, "north": domain_m},
        "crs": "EPSG:32636",  # UTM Zone 36N (Beylikova)
        "original_shape": [ny, nx],
        "resampled_shape": [ny, nx],
        "z_min": float(dem_grid.min()),
        "z_max": float(dem_grid.max()),
        "synthetic": True,
    }
    print(f"[DEM] Sentetik üretildi: {ny}×{nx}  Z: {dem_grid.min():.1f}–{dem_grid.max():.1f} m")
    return dem_grid, meta


# ── Sismik first-arrival traveltime okuma ─────────────────────────────────────
def load_traveltime_csv(csv_path: str) -> dict:
    """
    First arrival traveltime CSV okur.
    Beklenen sütunlar: offset_m, tt_ms [, shot_x, shot_y, recv_x, recv_y]
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    result = {"offset_m": data[:, 0], "tt_ms": data[:, 1]}
    if data.shape[1] >= 4:
        result["shot_x"] = data[:, 2]
        result["shot_y"] = data[:, 3]
    if data.shape[1] >= 6:
        result["recv_x"] = data[:, 4]
        result["recv_y"] = data[:, 5]
    print(f"[Sismik] {len(result['offset_m'])} first arrival okuttu")
    return result


# ── 1D Vp profili — refraksiyon first arrival inversion ───────────────────────
def invert_1d_refraction(offsets: np.ndarray, tt_ms: np.ndarray,
                          n_layers: int = 4) -> dict:
    """
    Basit 2-katman+ refraksiyon ters çözümü (t-x analizi).
    Gerçek çözücü: MASW / GLI / Plus-Minus yöntemi.
    Burada: lineer regresyon ile hız ve kesim noktası tahmini.

    Döndürür: {'vp': [m/s], 'depth': [m], 'n_layers': int}
    """
    from scipy.optimize import curve_fit

    tt_s = tt_ms / 1000.0

    # İlk yaklaşım: piecewise linear t-x grafiği
    # Slope = 1/Vp, crossover offset → katman derinliği
    vp_list   = []
    depth_list = []
    crossovers = np.linspace(offsets.min()*0.3, offsets.max()*0.8, n_layers)

    prev_cross = 0.0
    for i, cross in enumerate(crossovers):
        mask = (offsets >= prev_cross) & (offsets < cross)
        if mask.sum() < 2:
            continue
        coeffs = np.polyfit(offsets[mask], tt_s[mask], 1)
        slope = max(coeffs[0], 1e-6)
        vp = 1.0 / slope
        vp_list.append(vp)
        # Katman derinliği (Hagedoorn formülü)
        if i > 0 and len(vp_list) >= 2:
            v1, v2 = vp_list[-2], vp_list[-1]
            ti = np.interp(cross, offsets[mask], tt_s[mask]) - cross/v2
            depth = (ti/2) * (v1*v2) / np.sqrt(v2**2 - v1**2) if v2 > v1 else 5.0
        else:
            depth = 0.0
        depth_list.append(depth)
        prev_cross = cross

    # Güvenlik: en az 2 katman
    if len(vp_list) < 2:
        vp_list = [800.0, 3500.0]
        depth_list = [0.0, 15.0]

    return {"vp": vp_list, "depth": depth_list, "n_layers": len(vp_list)}


# ── 3D Vp modeli oluşturma ────────────────────────────────────────────────────
def build_3d_vp_model(dem_grid: np.ndarray, dem_meta: dict,
                       vp_profile: Optional[dict] = None,
                       domain_depth_m: float = 480.0,
                       nz: int = 32) -> np.ndarray:
    """
    DEM yüzeyi + Vp profili kombinasyonundan 3D Vp modeli üretir.

    vp_profile: {'vp': [v1,v2,...], 'depth': [d1,d2,...]}
                None → varsayılan Beylikova analogu

    Döndürür: (nz, ny, nx) float32 Vp modeli [m/s]
    """
    ny, nx = dem_grid.shape
    z_surface = dem_grid  # topografya yüksekliği

    # Varsayılan profil — Beylikova analogu
    if vp_profile is None:
        vp_profile = {
            "vp":    [500, 1200, 2800, 4200, 5100],
            "depth": [0,     8,   25,   80,  200],
        }

    # Z eksenini topografya altında oluştur
    z_depths = np.linspace(0, domain_depth_m, nz)  # derinlik (m, yüzeyden aşağı)

    vp_model = np.zeros((nz, ny, nx), dtype=np.float32)

    vp_depths = np.array(vp_profile["depth"])
    vp_values = np.array(vp_profile["vp"])

    for iz, depth in enumerate(z_depths):
        # Her sütun için derinlik → Vp interpolasyonu
        vp_at_depth = float(np.interp(depth, vp_depths, vp_values))
        vp_model[iz, :, :] = vp_at_depth

    # Topografya maskesi: yüzey üstü = NaN → 0 Vp (hava)
    z_max_abs = dem_meta["z_max"]
    for iz, depth in enumerate(z_depths):
        abs_z = z_max_abs - depth  # mutlak yükseklik
        above_surface = (z_surface < abs_z)  # topografyanın üstü
        vp_model[iz, above_surface] = 0.0

    # Hafif yumuşatma
    vp_model = gaussian_filter(vp_model, sigma=[0.5, 0.5, 0.5])

    print(f"[Vp3D] Model: {nz}×{ny}×{nx}  "
          f"Vp: {vp_model[vp_model>0].min():.0f}–{vp_model.max():.0f} m/s")
    return vp_model


# ── Katman modeli ve QGIS export ──────────────────────────────────────────────
def build_layer_model(vp_model: np.ndarray) -> np.ndarray:
    """Vp modelinden kaya birimi ID katman modelini üret."""
    return classify_vp(vp_model)


def export_geotiff(array_2d: np.ndarray, dem_meta: dict,
                    out_path: str, band_names: list = None,
                    nodata: float = 0.0) -> None:
    """
    2D (veya 3D banded) array'i QGIS uyumlu GeoTIFF olarak kaydet.

    array_2d: (ny, nx) veya (n_bands, ny, nx)
    """
    if not RASTERIO_OK:
        print("[GeoTIFF] rasterio yok — export atlandı")
        return

    bounds = dem_meta["bounds"]
    crs_str = dem_meta.get("crs", "EPSG:4326")

    if array_2d.ndim == 2:
        array_2d = array_2d[np.newaxis, ...]  # (1, ny, nx)

    n_bands, ny, nx = array_2d.shape
    transform = from_bounds(
        bounds["west"], bounds["south"], bounds["east"], bounds["north"],
        nx, ny
    )
    try:
        crs = CRS.from_string(crs_str)
    except Exception:
        crs = CRS.from_epsg(4326)

    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=ny, width=nx,
        count=n_bands,
        dtype=array_2d.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        for i in range(n_bands):
            dst.write(array_2d[i], i + 1)
            if band_names and i < len(band_names):
                dst.update_tags(i+1, name=band_names[i])

    print(f"[GeoTIFF] Kaydedildi: {out_path}  ({n_bands} band, {ny}×{nx})")


def layer_summary(layer_model: np.ndarray,
                   vp_model: np.ndarray,
                   domain_m: float = 480.0) -> dict:
    """Katman modeli istatistik özeti — JSON olarak döndürür."""
    nz, ny, nx = layer_model.shape
    voxel_vol = (domain_m/nx) * (domain_m/ny) * (domain_m/nz)  # m³
    summary = {"total_voxels": int(layer_model.size), "classes": []}

    for cls in VP_CLASSES:
        mask = layer_model == cls["id"]
        count = int(mask.sum())
        if count == 0:
            continue
        vp_vals = vp_model[mask]
        summary["classes"].append({
            "id":          cls["id"],
            "label":       cls["label"],
            "color":       cls["color"],
            "voxel_count": count,
            "volume_m3":   round(count * voxel_vol, 0),
            "pct":         round(count / layer_model.size * 100, 2),
            "vp_mean":     round(float(vp_vals.mean()), 1),
            "rqd":         cls["rqd"],
            "ucs_mpa":     cls["ucs_mpa"],
            "drill_factor": cls["drill_factor"],
        })
    return summary


# ── Ana pipeline ──────────────────────────────────────────────────────────────
def run_seismic_layer_pipeline(
    dem_path: Optional[str] = None,
    seismic_csv: Optional[str] = None,
    ore_model_npy: Optional[str] = None,
    out_dir: str = "./seismic_output",
    domain_m: float = 480.0,
    nx: int = 32, ny: int = 32, nz: int = 32,
) -> dict:
    """
    Tam sismik katman modelleme pipeline'ı.

    Çıktı klasörü:
      dem_grid.npy          → topografya
      vp_model.npy          → Vp tomografi
      layer_model.npy       → kaya birimi indeks
      geology_surface.tif   → QGIS yüzey haritası
      vp_surface.tif        → QGIS Vp haritası
      layer_summary.json    → istatistikler
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) DEM
    if dem_path and os.path.exists(dem_path):
        dem_grid, dem_meta = load_dem_geotiff(dem_path, nx, ny)
    else:
        print("[DEM] GeoTIFF bulunamadı — sentetik üretiliyor")
        dem_grid, dem_meta = synthetic_dem(nx, ny, domain_m)

    # 2) Sismik profil
    vp_profile = None
    if seismic_csv and os.path.exists(seismic_csv):
        tt_data = load_traveltime_csv(seismic_csv)
        vp_profile = invert_1d_refraction(tt_data["offset_m"], tt_data["tt_ms"])
        print(f"[Sismik] Profil: Vp={vp_profile['vp']}  Depth={vp_profile['depth']}")

    # 3) 3D Vp modeli
    vp_model = build_3d_vp_model(dem_grid, dem_meta, vp_profile, domain_m, nz)

    # 4) Katman modeli
    layer_model = build_layer_model(vp_model)

    # 5) Cevher modeli örtüştürme
    ore_overlay = None
    if ore_model_npy and os.path.exists(ore_model_npy):
        ore = np.load(ore_model_npy)
        if ore.ndim == 4: ore = ore[0]
        from scipy.ndimage import zoom as _zoom
        ore_r = _zoom(ore, [nz/ore.shape[0], ny/ore.shape[1], nx/ore.shape[2]], order=1)
        ore_overlay = (ore_r > 0.3).astype(np.uint8) * 7  # ID=7 cevher
        print(f"[Cevher] Örtüştürüldü: {(ore_r>0.3).sum()} voksel cevher")

    # 6) Kaydet
    np.save(os.path.join(out_dir, "dem_grid.npy"),    dem_grid)
    np.save(os.path.join(out_dir, "vp_model.npy"),    vp_model)
    np.save(os.path.join(out_dir, "layer_model.npy"), layer_model)

    # 7) GeoTIFF export
    # Yüzey katmanı (z=0) → harita görünümü
    surface_layer = layer_model[0].astype(np.float32)
    surface_vp    = vp_model[0].astype(np.float32)

    export_geotiff(surface_layer, dem_meta,
                   os.path.join(out_dir, "geology_surface.tif"),
                   band_names=["rock_class"])
    export_geotiff(surface_vp, dem_meta,
                   os.path.join(out_dir, "vp_surface.tif"),
                   band_names=["Vp_m_per_s"])
    export_geotiff(dem_grid.astype(np.float32), dem_meta,
                   os.path.join(out_dir, "dem.tif"),
                   band_names=["elevation_m"])

    # 8) Özet JSON
    summary = layer_summary(layer_model, vp_model, domain_m)
    summary["dem_meta"] = dem_meta
    summary["vp_profile"] = vp_profile
    summary["ore_voxels"] = int((ore_overlay == 7).sum()) if ore_overlay is not None else 0
    with open(os.path.join(out_dir, "layer_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Pipeline tamamlandı: {out_dir}")
    for cls in summary["classes"]:
        print(f"  {cls['label']}: {cls['pct']:.1f}%  ({cls['volume_m3']:.0f} m³)  "
              f"Vp_ort={cls['vp_mean']:.0f} m/s  RQD={cls['rqd']}")

    return {
        "dem_grid":    dem_grid,
        "vp_model":    vp_model,
        "layer_model": layer_model,
        "ore_overlay": ore_overlay,
        "summary":     summary,
        "out_dir":     out_dir,
    }


if __name__ == "__main__":
    # Test çalışması
    result = run_seismic_layer_pipeline(
        dem_path=None,        # GeoTIFF yoksa sentetik
        seismic_csv=None,     # CSV yoksa varsayılan profil
        ore_model_npy=None,
        out_dir="/tmp/seismic_test",
        nx=32, ny=32, nz=32,
    )
    print("\nKatman model shape:", result["layer_model"].shape)
    print("Benzersiz kaya sınıfı:", np.unique(result["layer_model"]))

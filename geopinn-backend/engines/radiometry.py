"""
radiometry.py — Radyometri İleri Yönlü Modelleme & Veri İşleme
GeoPINN Studio 3.0

Desteklenen veri formatları:
    .csv / .xyz  : x, y, [z], U_ppm, Th_ppm, K_pct, [TC_cps], [dose_nGy/h]
    .dat         : boşluk/tab ayrımlı saha verisi
    .npy         : GeoPINN model grid (zaten radyometri konsantrasyonu)
    GeoSoft .grd : ileride

Fizik:
    Gammaray sayımı (cps), yüzey konsantrasyonundan derinlik düzeltmesiyle:

    cps_toplam ≈ c_U·U_ppm + c_Th·Th_ppm + c_K·K_pct + background

    Derinlik düzeltmesi (üstel zayıflama):
        signal(z) = signal_surface · exp(-μ · z)
        μ ≈ 0.0046 cm⁻¹ (granit için gammaray attenuation katsayısı)

    Thorium/Uranium oranı (Th/U):
        < 3.5 : taze magmatik → daha az alterasyon
        3.5-6 : orta alterasyon
        > 6   : ileri alterasyon / REE mobilizasyonu — hedef!

Referanslar:
    IAEA (2003) — Radioelement mapping guidelines
    Dickson & Scott (1997) — Interpretation of aerial gamma-ray surveys
"""

import numpy as np
from typing import Optional, Tuple, Dict


# ── Dönüşüm katsayıları (IAEA 2003, havadan gammaray için kalibrasyon) ─────────
# cps → ppm veya % (tipik detektor geometrisi için)
CPS_PER_PPM_U   = 1.505   # cps/ppm (NASVD kalibrasyon pedi)
CPS_PER_PPM_TH  = 0.623   # cps/ppm
CPS_PER_PCT_K   = 11.93   # cps/%

# Gammaray attenuation (granit/toprak, ~1 MeV ortalama enerji)
MU_ROCK_PER_M   = 0.046   # m⁻¹ (0.0046 cm⁻¹ → 0.046 m⁻¹)

# eU (uranyum eşdeğeri Th için)
TH_TO_EU_FACTOR = 0.3348  # eU_ppm = Th_ppm * 0.3348 (Th/U denge varsayımı)


# ── Radyometri ileri model ────────────────────────────────────────────────────
def forward_radiometry_surface(
    u_ppm:  np.ndarray,
    th_ppm: np.ndarray,
    k_pct:  np.ndarray,
    background_cps: float = 15.0,
) -> Dict[str, np.ndarray]:
    """
    U/Th/K konsantrasyonlarından beklenen yüzey gammaray sayımını hesaplar.

    Parametreler:
        u_ppm, th_ppm, k_pct : (nx, ny) veya (nx, ny, nz) yüzey/3D grid
        background_cps       : kozmik + cihaz arka plan sayımı

    Döndürür: dict
        TC_cps   : Toplam sayım (Total Count) [cps]
        U_cps    : Uranyum kanalı
        Th_cps   : Toryum kanalı
        K_cps    : Potasyum kanalı
        eU_ppm   : Uranyum eşdeğeri (Th dahil)
        Th_U     : Th/U oranı (alterasyon indikatörü)
        dose_nGy_h : Absorbe doz hızı [nGy/h]
    """
    # 3D ise yüzeyi al (iz=0)
    if u_ppm.ndim == 3:
        u2  = u_ppm[:, :, 0]
        th2 = th_ppm[:, :, 0]
        k2  = k_pct[:, :, 0]
    else:
        u2, th2, k2 = u_ppm, th_ppm, k_pct

    u_ch  = CPS_PER_PPM_U  * u2
    th_ch = CPS_PER_PPM_TH * th2
    k_ch  = CPS_PER_PCT_K  * k2
    tc    = u_ch + th_ch + k_ch + background_cps

    # Absorbe doz (IAEA 2003 katsayıları, nGy/h)
    dose = 0.462 * k2*10000 + 0.604 * u2*1000 + 0.287 * th2*1000  # basit lineer model
    # Tam formül: dose [nGy/h] = 1.505*K[%] + 0.653*eU[ppm] + 0.287*Th[ppm]
    dose = 1.505 * k2 + 0.653 * (u2 + TH_TO_EU_FACTOR*th2) + 0.287 * th2

    # Th/U oranı (sıfırdan kaçın)
    th_u = np.where(u2 > 0.01, th2 / u2, np.nan)

    # eU
    eu = u2 + TH_TO_EU_FACTOR * th2

    return {
        "TC_cps":     tc,
        "U_cps":      u_ch,
        "Th_cps":     th_ch,
        "K_cps":      k_ch,
        "eU_ppm":     eu,
        "Th_U":       th_u,
        "dose_nGy_h": dose,
    }


def depth_correction(
    surface_signal: np.ndarray,
    depth_m: float,
    mu: float = MU_ROCK_PER_M,
) -> np.ndarray:
    """
    Derinlikten gelen sinyal zayıflamasını hesaplar.
    Saha ölçümü yüzeyde, hedef derinlikte → sinyal bu kadar zayıflamış.
    """
    return surface_signal * np.exp(-mu * depth_m)


def ree_alteration_index(
    u_ppm:  np.ndarray,
    th_ppm: np.ndarray,
    k_pct:  np.ndarray,
    la_ppm: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    REE yatağı arama için radyometri tabanlı alterasyon indeksleri.

    İndeksler:
        Th/U   : > 4 → gelişmiş alterasyon, REE mobilizasyonu
        F-parametre : (Th + 3.48*K_ppm) / U — CARLIN tipi Au ile ayırt için
        K/Th   : potasyum zenginleşmesi (K-feldispat alterasyonu)
        eU     : efektif uranyum (monazit/xenotim indikatörü)

    Döndürür: dict of 2D anomali haritaları
    """
    u2  = u_ppm[:, :, 0]  if u_ppm.ndim  == 3 else u_ppm
    th2 = th_ppm[:, :, 0] if th_ppm.ndim == 3 else th_ppm
    k2  = k_pct[:, :, 0]  if k_pct.ndim  == 3 else k_pct

    eps = 1e-6
    th_u   = np.where(u2 > eps, th2 / u2, np.nan)
    k_th   = np.where(th2 > eps, k2 / th2, np.nan)
    eu     = u2 + TH_TO_EU_FACTOR * th2
    f_par  = np.where(u2 > eps, (th2 + 3.48 * k2 * 10000 / 1e6) / u2, np.nan)  # K %→ppm

    # Anomali eşikleme (Beylikova için tipik arka plan değerleri)
    U_BACKGROUND  = 3.0   # ppm
    TH_BACKGROUND = 12.0  # ppm
    K_BACKGROUND  = 2.5   # %

    u_anomaly  = np.clip((u2  - U_BACKGROUND)  / U_BACKGROUND,  0, None)
    th_anomaly = np.clip((th2 - TH_BACKGROUND) / TH_BACKGROUND, 0, None)
    k_anomaly  = np.clip((k2  - K_BACKGROUND)  / K_BACKGROUND,  0, None)

    # Bileşik anomali skoru (0-1 arası normalize)
    composite = (u_anomaly + th_anomaly * 1.5 + k_anomaly * 0.5) / 3.0
    composite = np.clip(composite / (composite.max() + 1e-9), 0, 1)

    result = {
        "Th_U_ratio":       th_u,
        "K_Th_ratio":       k_th,
        "eU_ppm":           eu,
        "F_parameter":      f_par,
        "U_anomaly":        u_anomaly,
        "Th_anomaly":       th_anomaly,
        "K_anomaly":        k_anomaly,
        "composite_score":  composite,
        "ree_target_prob":  _ree_probability(th_u, eu, k2),
    }
    if la_ppm is not None:
        la2 = la_ppm[:, :, 0] if la_ppm.ndim == 3 else la_ppm
        result["La_ppm"] = la2
    return result


def _ree_probability(th_u, eu, k_pct) -> np.ndarray:
    """
    Basit buluşsal (heuristic) REE yatağı olasılık skoru.
    Gerçek fuzzy logic veya makine öğrenmesi ile değiştirilebilir.

    Kural seti (Beylikova analogu için):
        1. Th/U > 4 ve eU > 5 ppm → yüksek olasılık
        2. K/% > 3.5 ve Th/U > 3 → orta-yüksek
        3. eU > 8 ppm             → yüksek
    """
    score = np.zeros_like(eu, dtype=float)
    score += np.where(~np.isnan(th_u) & (th_u > 4) & (eu > 5), 0.4, 0)
    score += np.where(~np.isnan(th_u) & (th_u > 3) & (k_pct > 3.5), 0.25, 0)
    score += np.where(eu > 8, 0.3, 0)
    score += np.where(eu > 12, 0.2, 0)  # çift sayma kasıtlı — çok güçlü anomali
    return np.clip(score, 0, 1)


# ── Veri formatı işleyicileri ─────────────────────────────────────────────────
SUPPORTED_FORMATS = {
    "radiometry_csv": {
        "extensions": [".csv", ".xyz", ".dat", ".txt"],
        "description": "Saha radyometri verisi (x, y, U_ppm, Th_ppm, K_pct)",
        "columns": ["x", "y", "U_ppm", "Th_ppm", "K_pct"],
        "optional": ["z", "TC_cps", "dose_nGy_h", "La_ppm", "Ce_ppm"],
    },
    "sp_csv": {
        "extensions": [".csv", ".dat", ".txt"],
        "description": "Öz-potansiyel verisi (x, y, SP_mV)",
        "columns": ["x", "y", "SP_mV"],
        "optional": ["z", "electrode_type", "survey_line"],
    },
    "grav_csv": {
        "extensions": [".csv", ".dat", ".txt"],
        "description": "Gravite saha verisi (x, y, gz_mGal)",
        "columns": ["x", "y", "gz_mGal"],
        "optional": ["z", "Bouguer_mGal", "FAA_mGal", "terrain_corr"],
    },
    "mag_csv": {
        "extensions": [".csv", ".dat", ".txt"],
        "description": "Manyetik alan verisi (x, y, TMI_nT)",
        "columns": ["x", "y", "TMI_nT"],
        "optional": ["z", "RTP_nT", "tilt_angle", "analytic_signal"],
    },
    "ip_csv": {
        "extensions": [".csv", ".dat"],
        "description": "Indüklenmiş Polarizasyon (IP/Res dipole-dipole)",
        "columns": ["x_mid", "depth", "chargeability_ms", "resistivity_ohmm"],
        "optional": ["phase_mrad", "n_spacing", "array_type"],
    },
    "seismic_refrac_csv": {
        "extensions": [".csv", ".dat"],
        "description": "Sismik refraksiyon (offset-traveltime)",
        "columns": ["offset_m", "tt_ms"],
        "optional": ["shot_x", "shot_y", "recv_x", "recv_y"],
    },
    "gpr_csv": {
        "extensions": [".csv", ".dat"],
        "description": "GPR profil verisi (mesafe, zaman)",
        "columns": ["distance_m", "twt_ns"],
        "optional": ["amplitude", "frequency_MHz"],
    },
    "model_npy": {
        "extensions": [".npy"],
        "description": "GeoPINN 3D model grid (numpy array)",
        "columns": [],
        "optional": [],
    },
}


def detect_format(filename: str, header_preview: str = "") -> str:
    """
    Dosya adı ve başlık önizlemesinden format tahmin eder.
    Döndürür: SUPPORTED_FORMATS anahtarı veya 'unknown'
    """
    import os
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".npy":
        return "model_npy"

    header_lower = header_preview.lower()

    # Radyometri belirteçleri
    if any(k in header_lower for k in ['u_ppm','th_ppm','thorium','uranium','k_pct',
                                         'potassium','tc_cps','total_count','dose']):
        return "radiometry_csv"

    # SP belirteçleri
    if any(k in header_lower for k in ['sp_mv','self_pot','sp ','spontaneous']):
        return "sp_csv"

    # IP belirteçleri
    if any(k in header_lower for k in ['chargeability','ip ','ms','mrad','phase']):
        return "ip_csv"

    # Sismik belirteçleri
    if any(k in header_lower for k in ['tt_ms','traveltime','offset_m','shot']):
        return "seismic_refrac_csv"

    # GPR belirteçleri
    if any(k in header_lower for k in ['twt_ns','two_way','gpr','trace']):
        return "gpr_csv"

    # Manyetik belirteçleri
    if any(k in header_lower for k in ['tmi','nT','tmi_nt','rtp','mag ']):
        return "mag_csv"

    # Gravite belirteçleri
    if any(k in header_lower for k in ['mgal','bouguer','faa','gz_mgal']):
        return "grav_csv"

    return "unknown"


def read_field_data(path: str, format_key: str = None) -> Tuple[dict, str]:
    """
    Saha verisini okur, format'a göre doğru parser'ı seçer.

    Döndürür: (data_dict, detected_format)
    """
    import os
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        return {"array": arr, "shape": list(arr.shape)}, "model_npy"

    # Başlık önizlemesi ile format tespiti
    try:
        with open(path, encoding='utf-8-sig', errors='replace') as f:
            header = f.read(1024)
    except Exception:
        header = ""

    if format_key is None:
        format_key = detect_format(path, header)

    # Genel CSV parser
    import csv
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(header, delimiters=',\t; ')
        sep = dialect.delimiter
    except Exception:
        sep = ','

    try:
        data = np.genfromtxt(path, delimiter=sep, names=True, dtype=None,
                              encoding='utf-8-sig', invalid_raise=False)
    except Exception as e:
        raise ValueError(f"Veri okunamadı: {e}")

    result = {}
    names = [n.lower() for n in data.dtype.names]

    # Format bilgisinden beklenen sütunları al
    fmt = SUPPORTED_FORMATS.get(format_key, {})
    all_cols = fmt.get("columns", []) + fmt.get("optional", [])

    for col in data.dtype.names:
        result[col.lower()] = data[col].astype(float)

    result["_format"] = format_key
    result["_n_points"] = len(data)
    return result, format_key


# ── İstatistik özeti ──────────────────────────────────────────────────────────
def radiometry_stats(data: dict) -> dict:
    """
    Radyometri verisinden temel istatistik ve anomali özeti üretir.
    """
    stats = {}
    for key in ["u_ppm", "th_ppm", "k_pct", "tc_cps", "dose_ngsy_h"]:
        if key in data:
            v = data[key]
            v_clean = v[np.isfinite(v)]
            if len(v_clean) > 0:
                stats[key] = {
                    "mean":   float(np.mean(v_clean)),
                    "median": float(np.median(v_clean)),
                    "std":    float(np.std(v_clean)),
                    "min":    float(np.min(v_clean)),
                    "max":    float(np.max(v_clean)),
                    "p90":    float(np.percentile(v_clean, 90)),
                }

    # Th/U oranı
    if "u_ppm" in data and "th_ppm" in data:
        u = data["u_ppm"]; th = data["th_ppm"]
        valid = (u > 0.1) & np.isfinite(u) & np.isfinite(th)
        if valid.sum() > 0:
            th_u = th[valid] / u[valid]
            stats["Th_U_ratio"] = {
                "mean": float(np.mean(th_u)),
                "median": float(np.median(th_u)),
                "max": float(np.max(th_u)),
                "anomaly_threshold": 4.0,
                "n_anomalous": int((th_u > 4.0).sum()),
                "interpretation": "Yüksek Th/U → gelişmiş alterasyon, REE taşınımı"
                                  if np.median(th_u) > 3.5 else "Normal Th/U",
            }

    return stats


if __name__ == "__main__":
    print("[TEST] Radyometri modülü test ediliyor...")

    # Sentetik 2D radyometri verisi
    nx, ny = 21, 21
    x = np.linspace(0, 480, nx)
    y = np.linspace(0, 480, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Arka plan + anomali
    u  = 3.0 + 9.0 * np.exp(-((X-240)**2 + (Y-240)**2) / (80**2))
    th = 12.0 + 43.0 * np.exp(-((X-240)**2 + (Y-240)**2) / (100**2))
    k  = 2.5 + 1.7 * np.exp(-((X-200)**2 + (Y-280)**2) / (120**2))

    result = forward_radiometry_surface(u, th, k)
    idx = ree_alteration_index(u, th, k)

    print(f"  TC maks: {result['TC_cps'].max():.1f} cps")
    print(f"  Th/U maks: {np.nanmax(idx['Th_U_ratio']):.2f}")
    print(f"  REE hedef skoru maks: {idx['ree_target_prob'].max():.3f}")
    print(f"  Yüksek olasılıklı piksel: {(idx['ree_target_prob']>0.5).sum()}")
    print("[TEST] Tamamlandı.")

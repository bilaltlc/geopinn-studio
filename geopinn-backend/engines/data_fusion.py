"""
data_fusion.py — Çok-Yöntemli Jeofizik Veri Füzyonu
GeoPINN Studio v3.1.0

Referanslar:
    Porwal et al. (2003) — Mineral potential mapping with mathematical geological
        models. Nat. Resour. Res., 12(1), 1-25.
    Carranza & Hale (2002) — Wildcat mapping of gold potential. Nat. Resour. Res.
    Bonham-Carter (1994) — Geographic Information Systems for Geoscientists.
    Krishnamurthy et al. (2020) — Integrated geophysical analysis for REE deposits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata


# ── Normalizasyon ─────────────────────────────────────────────────────────────
def minmax_normalize(arr: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    lo = arr.min() if vmin is None else vmin
    hi = arr.max() if vmax is None else vmax
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def rank_normalize(arr: np.ndarray) -> np.ndarray:
    flat = arr.ravel()
    ranked = (rankdata(flat) - 1) / (len(flat) - 1)
    return ranked.reshape(arr.shape)


# ── Tek yöntem anomali skoru ──────────────────────────────────────────────────
def gravity_anomaly_score(gz_mgal: np.ndarray) -> np.ndarray:
    """Bouguer anomali → pozitif normalize skor (yüksek yoğunluk = yüksek skor)."""
    return minmax_normalize(gz_mgal)


def magnetic_anomaly_score(tmi_nt: np.ndarray) -> np.ndarray:
    """TMI → normalize skor (abs değer — hem pozitif hem negatif anomali hedef)."""
    return minmax_normalize(np.abs(tmi_nt))


def csamt_anomaly_score(rho_a_ohmm: np.ndarray) -> np.ndarray:
    """Görünür özdirenç → düşük direnç = yüksek skor (iletken alterasyon)."""
    return minmax_normalize(-np.log10(np.clip(rho_a_ohmm, 0.1, 1e5)))


def ip_anomaly_score(chargeability: np.ndarray) -> np.ndarray:
    """Chargeability → yüksek = yüksek skor (sülfür mineralizasyonu)."""
    return minmax_normalize(chargeability)


def sp_anomaly_score(sp_mv: np.ndarray) -> np.ndarray:
    """SP [mV] → negatif anomali = yüksek skor (hidrotermal kaynak)."""
    return minmax_normalize(-sp_mv)


def radiometry_anomaly_score(
    th_u_ratio: np.ndarray,
    eu_ppm:     np.ndarray,
) -> np.ndarray:
    """Th/U + eU → REE alterasyon skoru."""
    th_u_norm = minmax_normalize(th_u_ratio)
    eu_norm   = minmax_normalize(eu_ppm)
    return 0.6 * th_u_norm + 0.4 * eu_norm


def heat_flow_anomaly_score(q_mw_m2: np.ndarray) -> np.ndarray:
    """Isı akışı [mW/m²] → yüksek = yüksek skor (hidrotermal aktivite)."""
    return minmax_normalize(q_mw_m2)


# ── Füzyon yöntemleri ─────────────────────────────────────────────────────────
def weighted_sum_fusion(
    scores: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """
    Ağırlıklı toplam füzyon (Bonham-Carter 1994).
    W_total = Σ(wᵢ · sᵢ) / Σwᵢ
    """
    total_w = sum(weights.get(k, 0) for k in scores)
    if total_w < 1e-12:
        return np.zeros_like(next(iter(scores.values())))
    result = np.zeros_like(next(iter(scores.values())), dtype=np.float64)
    for k, s in scores.items():
        w = weights.get(k, 0)
        result += w * s
    return result / total_w


def fuzzy_and_fusion(scores: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Fuzzy AND (minimum operatör) — Porwal et al. (2003).
    Tüm kanallarda eş zamanlı anomali gerektirir.
    Konservatif — yanlış pozitif az.
    """
    arrays = list(scores.values())
    result = arrays[0].copy()
    for a in arrays[1:]:
        result = np.minimum(result, a)
    return result


def fuzzy_or_fusion(scores: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Fuzzy OR (maksimum operatör).
    Herhangi bir kanalda anomali yeterli.
    Liberal — yanlış pozitif fazla.
    """
    arrays = list(scores.values())
    result = arrays[0].copy()
    for a in arrays[1:]:
        result = np.maximum(result, a)
    return result


def fuzzy_gamma_fusion(
    scores: Dict[str, np.ndarray],
    gamma: float = 0.85,
) -> np.ndarray:
    """
    Fuzzy GAMMA operatörü (Zimmermann & Zysno 1980).
    F_γ = (Fuzzy OR)^γ · (Fuzzy AND)^(1-γ)

    γ=1 → OR, γ=0 → AND, γ=0.85 → Bonham-Carter 1994 önerisi.
    Mineral potential mapping için standart.
    """
    f_or  = fuzzy_or_fusion(scores)
    f_and = fuzzy_and_fusion(scores)
    return (f_or ** gamma) * (f_and ** (1 - gamma))


def index_overlay_fusion(
    scores: Dict[str, np.ndarray],
    thresholds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Index overlay (sayım tabanlı) — Carranza & Hale (2002).
    Her kanal için eşiği aşan pikseller sayılır.
    Sonuç: 0..N arası tamsayı (N = kanal sayısı).
    """
    if thresholds is None:
        thresholds = {k: 0.5 for k in scores}
    result = np.zeros_like(next(iter(scores.values())), dtype=np.float64)
    for k, s in scores.items():
        thr = thresholds.get(k, 0.5)
        result += (s >= thr).astype(np.float64)
    return result / max(len(scores), 1)


# ── Ana füzyon motoru ─────────────────────────────────────────────────────────
class DataFusionEngine:
    """
    7 jeofizik yöntemden bileşik REE anomali skoru.

    Yöntemler:
        gravity      — Bouguer anomali (yoğunluk kontrastı)
        magnetic     — TMI (süseptibilite kontrastı)
        csamt        — Görünür özdirenç (alterasyon zonu)
        ip           — Chargeability (sülfür mineralizasyonu)
        sp           — Öz-potansiyel (hidrotermal akış)
        radiometry   — Th/U + eU (REE iz elementi)
        heat_flow    — Radyojenik ısı akışı (hidrotermal sistem)
    """

    # Beylikova REE arama için varsayılan ağırlıklar
    # (Krishnamurthy et al. 2020, REE deposit analogues)
    DEFAULT_WEIGHTS = {
        "gravity":    0.15,
        "magnetic":   0.15,
        "csamt":      0.15,
        "ip":         0.20,   # Sülfür — REE ile birliktelik yüksek
        "sp":         0.15,   # Hidrotermal sistem göstergesi
        "radiometry": 0.15,   # REE iz elementi
        "heat_flow":  0.05,
    }

    def fuse(
        self,
        # Giriş alanları — her biri (nx, ny) veya (nx, ny, nz) 2D/3D
        gravity_mgal:       Optional[np.ndarray] = None,
        magnetic_nt:        Optional[np.ndarray] = None,
        csamt_ohmm:         Optional[np.ndarray] = None,
        ip_chargeability:   Optional[np.ndarray] = None,
        sp_mv:              Optional[np.ndarray] = None,
        radiometry_th_u:    Optional[np.ndarray] = None,
        radiometry_eu_ppm:  Optional[np.ndarray] = None,
        heat_flow_mw_m2:    Optional[np.ndarray] = None,
        # Füzyon yöntemi
        method:   str = "gamma",    # "weighted" | "gamma" | "and" | "or" | "index"
        gamma:    float = 0.85,
        weights:  Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        # Çıktı grid boyutu (normalize için)
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """
        Mevcut yöntemlerin anomali skorlarını hesaplar ve birleştirir.

        En az 1 yöntem gereklidir. Eksik yöntemler otomatik dışlanır.

        Returns:
            composite_score  : (nx,ny) normalize [0..1] bileşik anomali
            method_scores    : her yöntemin ayrı skoru
            active_methods   : kullanılan yöntemler
            stats            : özet istatistikler
            interpretation   : jeolojik yorum
        """
        scores = {}
        ref_shape = None

        def _get_surface(arr):
            """3D array ise yüzey/maks projeksiyon al."""
            if arr is None:
                return None
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim == 3:
                a = a.max(axis=2)   # Maks projeksiyon (yüzey anomali)
            return a

        def _register(key, arr):
            nonlocal ref_shape
            if arr is None:
                return
            if ref_shape is None:
                ref_shape = arr.shape
            if arr.shape != ref_shape:
                from scipy.ndimage import zoom
                z = (ref_shape[0]/arr.shape[0], ref_shape[1]/arr.shape[1])
                arr = zoom(arr, z, order=1)
            scores[key] = arr

        # Anomali skorları
        grav = _get_surface(gravity_mgal)
        if grav is not None:
            _register("gravity", gravity_anomaly_score(grav))

        mag = _get_surface(magnetic_nt)
        if mag is not None:
            _register("magnetic", magnetic_anomaly_score(mag))

        csa = _get_surface(csamt_ohmm)
        if csa is not None:
            _register("csamt", csamt_anomaly_score(csa))

        ip_c = _get_surface(ip_chargeability)
        if ip_c is not None:
            _register("ip", ip_anomaly_score(ip_c))

        sp = _get_surface(sp_mv)
        if sp is not None:
            _register("sp", sp_anomaly_score(sp))

        if radiometry_th_u is not None and radiometry_eu_ppm is not None:
            th_u = _get_surface(radiometry_th_u)
            eu   = _get_surface(radiometry_eu_ppm)
            _register("radiometry", radiometry_anomaly_score(th_u, eu))
        elif radiometry_th_u is not None:
            th_u = _get_surface(radiometry_th_u)
            _register("radiometry", minmax_normalize(th_u))

        hf = _get_surface(heat_flow_mw_m2)
        if hf is not None:
            _register("heat_flow", heat_flow_anomaly_score(hf))

        if not scores:
            raise ValueError("En az 1 yöntem gerekli.")

        # Ağırlıklar
        w = weights if weights else {
            k: self.DEFAULT_WEIGHTS.get(k, 0.1) for k in scores
        }

        # Füzyon
        if method == "weighted":
            composite = weighted_sum_fusion(scores, w)
        elif method == "gamma":
            composite = fuzzy_gamma_fusion(scores, gamma)
        elif method == "and":
            composite = fuzzy_and_fusion(scores)
        elif method == "or":
            composite = fuzzy_or_fusion(scores)
        elif method == "index":
            composite = index_overlay_fusion(scores, thresholds)
        else:
            composite = fuzzy_gamma_fusion(scores, gamma)

        # Gaussian smoothing
        composite = gaussian_filter(composite, sigma=0.8)
        composite = minmax_normalize(composite)

        # Yüksek öncelikli hedef bölgeler
        thr_high = 0.75
        thr_med  = 0.50
        n_total  = composite.size
        n_high   = int((composite >= thr_high).sum())
        n_med    = int((composite >= thr_med).sum())

        # Korelasyon matrisi
        keys = list(scores.keys())
        corr = {}
        for i, k1 in enumerate(keys):
            for k2 in keys[i+1:]:
                c = float(np.corrcoef(
                    scores[k1].ravel(), scores[k2].ravel()
                )[0, 1])
                corr[f"{k1}↔{k2}"] = round(c, 3)

        stats = {
            "composite_max":    float(composite.max()),
            "composite_mean":   float(composite.mean()),
            "n_high_priority":  n_high,
            "n_medium_priority":n_med,
            "pct_high":         round(100 * n_high / n_total, 1),
            "pct_medium":       round(100 * n_med  / n_total, 1),
            "active_methods":   list(scores.keys()),
            "n_methods":        len(scores),
            "fusion_method":    method,
            "gamma":            gamma if method == "gamma" else None,
            "method_correlations": corr,
            "interpretation":   _interpret_composite(
                float(composite.max()), len(scores)
            ),
            "priority_zones":   _priority_zones(composite, thr_high, thr_med),
        }

        return {
            "composite_score": composite.tolist(),
            "method_scores":   {k: v.tolist() for k, v in scores.items()},
            "stats":           stats,
        }


def _interpret_composite(score_max: float, n_methods: int) -> str:
    if score_max < 0.4:
        return f"Düşük anomali ({n_methods} yöntem) — hedef olası değil"
    elif score_max < 0.6:
        return f"Orta anomali ({n_methods} yöntem) — potansiyel hedef, ek çalışma gerekli"
    elif score_max < 0.8:
        return f"Güçlü anomali ({n_methods} yöntem) — öncelikli hedef bölge"
    else:
        return f"Çok güçlü anomali ({n_methods} yöntem) — yüksek öncelikli sondaj hedefi"


def _priority_zones(
    composite: np.ndarray,
    thr_high:  float = 0.75,
    thr_med:   float = 0.50,
) -> List[Dict]:
    """Yüksek öncelikli kümeleri listeler."""
    from scipy.ndimage import label
    zones = []
    labeled, n = label(composite >= thr_high)
    for i in range(1, min(n + 1, 10)):
        mask = labeled == i
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        zones.append({
            "zone_id":    i,
            "priority":   "HIGH",
            "n_pixels":   int(mask.sum()),
            "max_score":  float(composite[mask].max()),
            "centroid_x": int(xs.mean()),
            "centroid_y": int(ys.mean()),
            "x_range":    [int(xs.min()), int(xs.max())],
            "y_range":    [int(ys.min()), int(ys.max())],
        })
    return sorted(zones, key=lambda z: -z["max_score"])[:5]


if __name__ == "__main__":
    print("[TEST] DataFusion — 7 yöntem bileşik anomali skoru")
    nx, ny = 32, 32
    rng = np.random.default_rng(42)

    def blob(cx, cy, r=6):
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        return np.exp(-((X-cx)**2+(Y-cy)**2)/(2*r**2))

    eng = DataFusionEngine()
    result = eng.fuse(
        gravity_mgal      = blob(16,16)*5 + rng.normal(0,0.5,(nx,ny)),
        magnetic_nt       = blob(16,16)*80 + rng.normal(0,5,(nx,ny)),
        csamt_ohmm        = 500 - blob(16,16)*480,
        ip_chargeability  = blob(16,16)*0.5,
        sp_mv             = -blob(16,16)*150,
        radiometry_th_u   = blob(16,16)*6 + 2,
        radiometry_eu_ppm = blob(16,16)*10 + 3,
        heat_flow_mw_m2   = blob(16,16)*40 + 60,
        method="gamma", gamma=0.85,
    )
    s = result["stats"]
    print(f"  Bileşik maks:    {s['composite_max']:.3f}")
    print(f"  Yüksek öncelik:  {s['n_high_priority']} piksel (%{s['pct_high']})")
    print(f"  Aktif yöntemler: {s['active_methods']}")
    print(f"  Yorum: {s['interpretation']}")
    if s['priority_zones']:
        z = s['priority_zones'][0]
        print(f"  1. Hedef bölge: merkez=({z['centroid_x']},{z['centroid_y']}), skor={z['max_score']:.3f}")
    print("[TEST] Tamamlandı.")

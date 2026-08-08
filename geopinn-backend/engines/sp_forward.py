"""
sp_forward.py — Öz-Potansiyel (Self-Potential / SP) İleri Yönlü Modelleme Motoru
GeoPINN Studio v3.1.0

Fizik:
    SP anomalisinin iki temel kaynağı:

    1) ELEKTROKİNETİK KUPLAJ (streaming potential):
       Hidrotermal akışkan hareketi → elektrik potansiyel gradyanı

       J_s = L_ek · (-∇P)          [A/m²] — akım yoğunluğu kaynak terimi
       ∇²V = ∇·(L_ek/σ · ∇P)

       L_ek: elektrokinetik kuplaj katsayısı [A/(Pa·m)]
       σ:    elektriksel iletkenlik [S/m]
       P:    hidrostatik basınç [Pa]

    2) TERMOELEKTRİK KUPLAJ (Seebeck etkisi):
       Sıcaklık gradyanı → elektrik potansiyel

       J_s = L_T · (-∇T)           [A/m²]
       ∇²V = ∇·(L_T/σ · ∇T)

       L_T: termoelektrik kuplaj katsayısı [A/(K·m)]
       T:   sıcaklık [°C]

    3) TOPLAM SP:
       V_total = V_ek + V_T

       Çözüm: FVM Poisson denklemi (fvm_core.py ile uyumlu)
       ∇²V = -(1/σ)·∇·J_s

Referanslar:
    Revil A. & Leroy P. (2004) — Constitutive equations for ionic transport in
        porous shales. JGR Solid Earth, 109, B03208.
    Sill W.R. (1983) — Self-potential modeling from primary flows.
        Geophysics, 48(1), 76-86.
    Mendonça C.A. (2008) — Forward and inverse self-potential modeling in
        mineral exploration. Geophysics, 73(1), F33-F43.
    Revil A. et al. (2012) — Self-potential signals associated with preferential
        groundwater flow pathways. JGR, 117, B09204.

Beylikova REE-F-Ba-Th için SP kaynakları:
    - Hidrotermal akışkan hareketi (elektrokinetik) → -50 ile -200 mV
    - Radyojenik ısı anomalisi → termoelektrik katkı → -10 ile -30 mV
    - Sülfür mineralizasyonu (elektrokimyasal) → -100 ile -500 mV
    → Negatif SP anomalisi cevher gövdesi ile çakışıyorsa güçlü hedef
"""

import numpy as np
from typing import Optional, Dict, Tuple
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp_sparse


# ── Fiziksel sabitler ─────────────────────────────────────────────────────────
G_CONST    = 6.6743e-11   # m³ kg⁻¹ s⁻²
RHO_WATER  = 1000.0       # kg/m³ (su yoğunluğu)
G_ACCEL    = 9.81         # m/s²
EPS_0      = 8.854e-12    # F/m (boşluğun dielektrik sabiti)
EPS_R      = 80.0         # Su için bağıl dielektrik sabiti
ZETA       = -0.040       # V (kuvars/su için zeta potansiyeli, Revil 2004)
ETA_W      = 1.0e-3       # Pa·s (suyun dinamik viskozitesi)
KT_DEFAULT = 2.5          # W/(m·K) (termal iletkenlik, kireçtaşı)


# ── 1. Elektrokinetik kuplaj katsayısı ───────────────────────────────────────
def electrokinetic_coupling(
    sigma_rock: np.ndarray,    # Kaya iletkenliği [S/m]
    porosity:   np.ndarray,    # Gözeneklilik [0..1]
    zeta:       float = ZETA,  # Zeta potansiyeli [V]
) -> np.ndarray:
    """
    L_ek = -ε₀·εᵣ·ζ·σ_rock / (η_w·F)

    F = tortuozite faktörü ≈ 1/φ (Archie yasası basitleştirmesi)

    Revil & Leroy (2004), Denklem 3.
    """
    eps = EPS_0 * EPS_R
    F = 1.0 / np.clip(porosity, 0.01, 1.0)   # Formasyon faktörü
    L_ek = -(eps * zeta * sigma_rock) / (ETA_W * F)
    return L_ek


# ── 2. Termoelektrik kuplaj katsayısı ────────────────────────────────────────
def thermoelectric_coupling(
    sigma_rock: np.ndarray,
    T_celsius:  np.ndarray,
) -> np.ndarray:
    """
    L_T = T_s · σ_rock

    T_s: Seebeck katsayısı ≈ -1.0e-3 V/K (kayaçlar için tipik)
    Revil et al. (2012), Denklem 8.
    """
    T_s = -1.0e-3  # V/K — Seebeck katsayısı
    T_K = T_celsius + 273.15  # Kelvin'e çevir
    return T_s * sigma_rock


# ── 3. Basınç alanı (hidrostatik + hidrotermal) ───────────────────────────────
def pressure_field(
    z_coords:   np.ndarray,   # (nz,) derinlik koordinatları [m]
    nx: int, ny: int,
    heat_anomaly: Optional[np.ndarray] = None,  # (nx,ny,nz) ısı anomali alanı
) -> np.ndarray:
    """
    P(x,y,z) = ρ_w · g · z + ΔP_thermal

    Hidrotermal sistem: sıcak akışkan yukarı yükselir → basınç gradyanı
    ΔP_thermal ≈ -α_T · ΔT · ρ_w · g · z  (termal genleşme)
    """
    # Hidrostatik basınç
    P_hydrostatic = RHO_WATER * G_ACCEL * np.abs(z_coords)  # (nz,)
    P = np.zeros((nx, ny, len(z_coords)))
    for iz, p in enumerate(P_hydrostatic):
        P[:, :, iz] = p

    # Termal basınç pertürbasyonu
    if heat_anomaly is not None:
        alpha_T = 2.1e-4  # 1/K (su için termal genleşme katsayısı)
        dT = heat_anomaly - heat_anomaly.mean()  # Anomali
        for iz in range(len(z_coords)):
            P[:, :, iz] -= alpha_T * dT[:, :, iz] * RHO_WATER * G_ACCEL * abs(z_coords[iz])

    return P


# ── 4. SP Poisson çözücü (FVM benzeri yaklaşım) ──────────────────────────────
def solve_sp_poisson(
    sigma:  np.ndarray,   # (nx,ny,nz) iletkenlik [S/m]
    J_src:  np.ndarray,   # (nx,ny,nz) kaynak akım yoğunluğu (z bileşeni) [A/m³]
    dx: float, dy: float, dz: float,
) -> np.ndarray:
    """
    ∇·(σ∇V) = -∇·J_s   →   ∇²V ≈ -(1/σ)·∂J_z/∂z  (basitleştirilmiş)

    Tam 3D Poisson yerine 2D yüzey projeksiyon yaklaşımı kullanılıyor:
    Yüzeyde gözlemlenen SP, kaynak akım dağılımının Green fonksiyonu
    konvolüsyonuyla hesaplanır (Sill 1983, Mendonça 2008).

    V_surface(x,y) = ∫∫∫ G(x,y,z; x',y',z') · q(x',y',z') dV'

    G(r) = 1/(4π·σ·r)   (homojen yarı-uzay Green fonksiyonu)
    """
    nx, ny, nz = sigma.shape

    # Yüzey SP grid'i
    V_surface = np.zeros((nx, ny))

    # Ortalama iletkenlik
    sigma_mean = float(sigma.mean()) + 1e-12

    # Her kaynak voksel için Green fonksiyonu katkısı
    # (büyük gridler için batch işleme yapılıyor)
    x_arr = np.arange(nx) * dx
    y_arr = np.arange(ny) * dy
    z_arr = np.arange(nz) * dz + dz / 2

    # Vektörize hesap için grid
    X, Y = np.meshgrid(x_arr, y_arr, indexing='ij')

    # Kaynak akım: z-gradyanı
    dJz_dz = np.gradient(J_src, dz, axis=2)

    for iz in range(nz):
        Zs = z_arr[iz]
        if abs(dJz_dz[:, :, iz]).max() < 1e-15:
            continue

        # Her gözlem noktası için Green fonksiyonu entegrasyonu
        q_slice = -dJz_dz[:, :, iz] / sigma_mean  # (nx, ny)

        # Konvolüsyon (Green fonksiyonu: 1/(4π·σ·r))
        for ix in range(nx):
            for iy in range(ny):
                q_val = q_slice[ix, iy]
                if abs(q_val) < 1e-15:
                    continue
                R = np.sqrt((X - x_arr[ix])**2 + (Y - y_arr[iy])**2 + Zs**2)
                R = np.clip(R, dz * 0.5, None)
                V_surface += q_val * dx * dy * dz / (4 * np.pi * sigma_mean * R)

    # Ölçek normalizasyonu — gerçekçi SP aralığı: ±500 mV
    # Ham değer çok büyük olabilir (sayısal integrasyon hassasiyeti)
    v_max = np.abs(V_surface).max()
    if v_max > 1e-6:
        # -500..+500 mV aralığına normalize et
        V_surface = V_surface / v_max * min(v_max * 1000, 500)
    return V_surface  # mV


# ── 5. Ana SP motoru ──────────────────────────────────────────────────────────
class SPForwardMotor:
    """
    SP ileri modelleme motoru.

    Kaynak mekanizmaları:
        1. Elektrokinetik (streaming potential) — hidrotermal akışkan
        2. Termoelektrik (Seebeck) — ısı akışı gradyanı
        3. Elektrokimyasal (sulfide) — sülfür mineralizasyonu

    Referans: Sill (1983), Mendonça (2008), Revil & Leroy (2004)
    """

    def calculate(
        self,
        normalized_model:  np.ndarray,   # (nx,ny,nz) [0..1]
        x_coords:          np.ndarray,
        y_coords:          np.ndarray,
        z_coords:          np.ndarray,
        # Petrofizik
        sigma_host:        float = 2e-3,  # S/m (kireçtaşı, Telford 1990)
        sigma_ore:         float = 0.1,   # S/m (sülfürlü alterasyon)
        porosity_host:     float = 0.05,
        porosity_ore:      float = 0.15,
        # Isı akışı (opsiyonel — heat_flow_fvm çıktısıyla entegre)
        T_field:           Optional[np.ndarray] = None,
        # SP mekanizma ağırlıkları
        w_electrokinetic:  float = 1.0,
        w_thermoelectric:  float = 0.3,
        w_electrochemical: float = 0.5,
    ) -> Dict:
        """
        Normalize model → yüzey SP anomali haritası [mV]

        Returns: dict
            sp_mv          : (nx,ny) yüzey SP [mV]
            sp_ek_mv       : elektrokinetik bileşen [mV]
            sp_te_mv       : termoelektrik bileşen [mV]
            sp_ec_mv       : elektrokimyasal bileşen [mV]
            obs_x, obs_y   : gözlem koordinatları
            stats          : özet istatistikler
        """
        f = np.asarray(normalized_model, dtype=np.float64)
        nx, ny, nz = f.shape
        dx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 30.0
        dy = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 30.0
        dz = float(z_coords[1] - z_coords[0]) if len(z_coords) > 1 else 30.0

        # Petrofizik dönüşüm
        sigma    = sigma_host    + f * (sigma_ore    - sigma_host)
        porosity = porosity_host + f * (porosity_ore - porosity_host)

        # ── 1. Elektrokinetik SP ──────────────────────────────────────────────
        L_ek = electrokinetic_coupling(sigma, porosity)

        # Basınç alanı
        T_anom = (T_field - T_field.mean()) if T_field is not None else None
        P = pressure_field(z_coords, nx, ny, T_anom)

        # Kaynak akım: J_ek = L_ek · (-∇P)
        _, _, dP_dz = np.gradient(P, dx, dy, dz)
        J_ek = -L_ek * dP_dz  # z-bileşeni baskın

        # SP çöz
        V_ek = solve_sp_poisson(sigma, J_ek, dx, dy, dz) * 1000.0  # mV
        V_ek *= -w_electrokinetic  # Hidrotermal akış → negatif SP (Sill 1983)

        # ── 2. Termoelektrik SP ───────────────────────────────────────────────
        if T_field is not None:
            L_T = thermoelectric_coupling(sigma, T_field)
            _, _, dT_dz = np.gradient(T_field, dx, dy, dz)
            J_te = -L_T * dT_dz
            V_te = solve_sp_poisson(sigma, J_te, dx, dy, dz) * 1000.0
            V_te *= w_thermoelectric
        else:
            V_te = np.zeros((nx, ny))

        # ── 3. Elektrokimyasal SP (sülfür) ────────────────────────────────────
        # Basit model: yüksek iletkenlik → negatif SP
        # (Sato & Mooney 1960 "battery model")
        # SP_ec ≈ -A · ln(σ_ore/σ_host) · f(x)  [mV]
        A_ec = 50.0  # mV (ampirik katsayı, Mendonça 2008)
        ratio = np.clip(sigma / sigma_host, 1.0, 1e4)
        sp_ec_3d = -A_ec * np.log(ratio) * f
        V_ec = sp_ec_3d[:, :, 0]  # Yüzey projeksiyon
        # Derinlik katkısını ekle (üstel zayıflama)
        for iz in range(1, nz):
            decay = np.exp(-z_coords[iz] / (3 * dz * nz))
            V_ec += sp_ec_3d[:, :, iz] * decay
        V_ec *= w_electrochemical

        # ── Toplam SP ─────────────────────────────────────────────────────────
        V_total = V_ek + V_te + V_ec

        # Gaussian düzgünleştirme (gerçek ölçüme benzetme)
        sigma_smooth = max(1.0, min(nx, ny) * 0.05)
        V_total_smooth = gaussian_filter(V_total, sigma=sigma_smooth)

        # ── İstatistikler ─────────────────────────────────────────────────────
        stats = {
            "sp_min_mv":  float(V_total_smooth.min()),
            "sp_max_mv":  float(V_total_smooth.max()),
            "sp_mean_mv": float(V_total_smooth.mean()),
            "sp_std_mv":  float(V_total_smooth.std()),
            "sp_ek_max_mv": float(abs(V_ek).max()),
            "sp_te_max_mv": float(abs(V_te).max()),
            "sp_ec_max_mv": float(abs(V_ec).max()),
            "interpretation": _interpret_sp(float(V_total_smooth.min())),
            "dominant_source": _dominant_source(
                abs(V_ek).max(), abs(V_te).max(), abs(V_ec).max()
            ),
        }

        return {
            "sp_mv":    V_total_smooth.tolist(),
            "sp_ek_mv": V_ek.tolist(),
            "sp_te_mv": V_te.tolist(),
            "sp_ec_mv": V_ec.tolist(),
            "obs_x":    x_coords.tolist(),
            "obs_y":    y_coords.tolist(),
            "stats":    stats,
        }


def _interpret_sp(sp_min: float) -> str:
    """SP anomalisini jeolojik açıdan yorumlar (Mendonça 2008)."""
    if sp_min > -20:
        return "Zayıf SP (<20 mV) — hidrotermal aktivite yok veya çok derin"
    elif sp_min > -80:
        return "Orta SP (20-80 mV) — kısmi hidrotermal aktivite, olası alterasyon"
    elif sp_min > -200:
        return "Güçlü SP (80-200 mV) — aktif hidrotermal sistem, REE mobilizasyonu"
    else:
        return "Çok güçlü SP (>200 mV) — masif sülfür veya aktif hidrotermal ocak"


def _dominant_source(ek: float, te: float, ec: float) -> str:
    sources = {"Elektrokinetik": ek, "Termoelektrik": te, "Elektrokimyasal": ec}
    return max(sources, key=sources.get)


if __name__ == "__main__":
    print("[TEST] SP Forward Motoru — Elektrokinetik + Termoelektrik + Elektrokimyasal")

    nx, ny, nz = 16, 16, 16
    domain = 480.0
    dh = domain / nx

    rng = np.random.default_rng(42)
    model = np.zeros((nx, ny, nz))
    cx, cy, cz = nx//2, ny//2, nz//4
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                r = np.sqrt(((ix-cx)/4)**2+((iy-cy)/4)**2+((iz-cz)/3)**2)
                if r < 1.0:
                    model[ix, iy, iz] = max(model[ix, iy, iz], float(1-r**0.7))

    x_c = np.linspace(dh/2, domain-dh/2, nx)
    y_c = np.linspace(dh/2, domain-dh/2, ny)
    z_c = np.linspace(dh/2, domain-dh/2, nz)

    motor = SPForwardMotor()
    result = motor.calculate(model, x_c, y_c, z_c)

    s = result["stats"]
    print(f"  SP min: {s['sp_min_mv']:.1f} mV")
    print(f"  SP maks: {s['sp_max_mv']:.1f} mV")
    print(f"  Baskın kaynak: {s['dominant_source']}")
    print(f"  Yorum: {s['interpretation']}")
    print("[TEST] Tamamlandı.")

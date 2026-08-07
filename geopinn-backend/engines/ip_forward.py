"""
ip_forward.py — Indüklenmiş Polarizasyon (IP) İleri Yönlü Modelleme Motoru
GeoPINN Studio v3.1.0

Fizik:
    Cole-Cole kompleks özdirenç modeli (Pelton et al., 1978):

        ρ*(ω) = ρ₀ · [1 - m · (1 - 1/(1 + (iωτ)^c))]

    Parametreler:
        ρ₀  : DC özdirenç [Ω·m]
        m   : dolabilirlik / chargeability [adimensional, 0..1]
        τ   : relaksasyon süresi [s]
        c   : frekans bağımlılık katsayısı [adimensional, 0..1]

    Görünür özdirenç ve faz hesabı:
        ρ_a(ω) = K · ΔV(ω) / I        (K = geometrik faktör)
        φ(ω)   = -Im(ρ*(ω)) / Re(ρ*(ω)) · 1000   [mrad]

    Dipole-Dipole dizaynı (standart IP saha ölçümü):
        C1 ─── C2 ─── P1 ─── P2   (a = elektrot aralığı, n = ayırma katsayısı)
        K = π·n·(n+1)·(n+2)·a     (Geometrik faktör, Telford et al., 1990)

    3D forward modelleme:
        Yüzey gözlem verisi, Born yaklaşımı ve Green fonksiyonu ile hesaplanır
        (Oldenburg & Li, 1994). GPU'da vektörize hesap.

Referanslar:
    Pelton W.H. et al. (1978) — Mineral discrimination and removal of inductive
        coupling with multifrequency IP. Geophysics, 43(3), 588-609.
    Oldenburg D.W. & Li Y. (1994) — Inversion of induced polarization data.
        Geophysics, 59(9), 1327-1341.
    Loke M.H. & Barker R.D. (1996) — Rapid least-squares inversion of apparent
        resistivity pseudosections by a quasi-Newton method. Geophysical Prospecting.
    Telford W.M. et al. (1990) — Applied Geophysics. Cambridge University Press.

Beylikova REE-F-Ba-Th için tipik IP parametreleri:
    Host kaya (kireçtaşı/şist):  ρ₀=500 Ω·m, m=0.02, τ=0.1s, c=0.5
    Sülfürlü alterasyon:          ρ₀=50  Ω·m, m=0.25, τ=0.5s, c=0.4
    Pirit/Arsenopirit zonu:       ρ₀=10  Ω·m, m=0.60, τ=1.0s, c=0.3
    → Yüksek m (>0.3) + düşük ρ₀ → güçlü IP anomalisi → sülfür/cevher hedefi
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict


class ColeColeIP:
    """
    Cole-Cole kompleks özdirenç modeli ile IP forward modelleme.
    PyTorch GPU hızlandırmalı, CSAMT motoru ile aynı altyapı.

    Kullanım:
        motor = ColeColeIP()
        sonuc = motor.forward_pseudosection(
            rho0_field, m_field, tau_field, c_field,
            x_coords, z_coords, obs_x, frequencies, a_spacing, n_max
        )
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BİLGİ] IP HESAPLAMA DONANIMI: {self.device}".upper())

    # ── Cole-Cole kompleks özdirenç ───────────────────────────────────────────
    def cole_cole(
        self,
        omega: torch.Tensor,      # Açısal frekans [rad/s], shape (N_freq,)
        rho0:  torch.Tensor,      # DC özdirenç [Ω·m], shape (N_cells,)
        m:     torch.Tensor,      # Chargeability [0..1], shape (N_cells,)
        tau:   torch.Tensor,      # Relaksasyon süresi [s], shape (N_cells,)
        c:     torch.Tensor,      # Frekans üsteli [0..1], shape (N_cells,)
    ) -> torch.Tensor:
        """
        Cole-Cole: ρ*(ω) = ρ₀·[1 - m·(1 - 1/(1+(iωτ)^c))]

        Returns: complex tensor, shape (N_cells, N_freq)
        """
        # Broadcasting: (N_cells, 1) × (1, N_freq)
        omega = omega.unsqueeze(0)      # (1, N_freq)
        rho0  = rho0.unsqueeze(-1)      # (N_cells, 1)
        m_    = m.unsqueeze(-1)         # (N_cells, 1)
        tau_  = tau.unsqueeze(-1)       # (N_cells, 1)
        c_    = c.unsqueeze(-1)         # (N_cells, 1)

        # iωτ kompleks sayı
        i_omega_tau = torch.complex(
            torch.zeros_like(omega * tau_),
            omega * tau_
        )  # shape (N_cells, N_freq)

        # (iωτ)^c = exp(c · ln(iωτ))
        # ln(iωτ) = ln|ωτ| + i·π/2  (ωτ > 0 için)
        omega_tau_abs = (omega * tau_).abs().clamp(min=1e-30)
        ln_abs = torch.log(omega_tau_abs)
        ln_iomega_tau = torch.complex(ln_abs, torch.full_like(ln_abs, torch.pi / 2))
        iomega_tau_c = torch.exp(c_ * ln_iomega_tau)  # (N_cells, N_freq) complex

        # Cole-Cole faktörü: 1 - m·(1 - 1/(1 + (iωτ)^c))
        one = torch.ones_like(iomega_tau_c)
        cc_factor = one - m_ * (one - one / (one + iomega_tau_c))

        return rho0 * cc_factor  # (N_cells, N_freq) complex

    # ── Geometrik faktör (dipole-dipole) ─────────────────────────────────────
    @staticmethod
    def dipole_dipole_K(a: float, n: int) -> float:
        """
        Dipole-dipole geometrik faktörü.
        K = π·n·(n+1)·(n+2)·a    [Telford et al., 1990]
        """
        return np.pi * n * (n + 1) * (n + 2) * a

    # ── 1D görünür özdirenç (layered earth) ──────────────────────────────────
    def apparent_resistivity_1d(
        self,
        rho_complex: torch.Tensor,   # (N_layers, N_freq) complex
        thicknesses: torch.Tensor,   # (N_layers-1,) real
        a_spacing:   float,
        n_factor:    int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1D layered earth için görünür özdirenç ve faz hesabı.
        Wait (1954) özyinelemesi — CSAMT motoruyla benzer yaklaşım.

        Returns:
            rho_a  : görünür özdirenç [Ω·m], shape (N_freq,)
            phase  : faz [mrad], shape (N_freq,)
        """
        N_layers, N_freq = rho_complex.shape
        h = thicknesses.to(self.device, dtype=torch.float64)

        # Pseudo-depth (Loke & Barker, 1996): z ≈ 0.519·n·a
        pseudo_depth = 0.519 * n_factor * a_spacing

        # Basit 1D görünür özdirenç: ağırlıklı ortalama
        # (Tam çözüm için Hankel transform gerekir — bu iteratif yaklaşım)
        depths = torch.zeros(N_layers, device=self.device, dtype=torch.float64)
        for i in range(1, N_layers):
            depths[i] = depths[i-1] + h[i-1] if i <= len(h) else depths[i-1] + 1000.0

        # Pseudo-depth'e katkı ağırlıkları (Gaussian kernel)
        sigma = pseudo_depth * 0.5
        weights = torch.exp(-((depths - pseudo_depth) ** 2) / (2 * sigma ** 2))
        weights = weights / weights.sum()  # normalize

        # Ağırlıklı kompleks özdirenç
        rho_a_complex = (weights.unsqueeze(-1) * rho_complex).sum(dim=0)  # (N_freq,)

        rho_a = rho_a_complex.real.clamp(min=1e-6)
        phase = -torch.atan2(rho_a_complex.imag, rho_a_complex.real) * 1000.0  # mrad

        return rho_a, phase

    # ── 3D forward: pseudosection üretimi ────────────────────────────────────
    def forward_pseudosection(
        self,
        rho0_3d:  np.ndarray,    # (nx, ny, nz) DC özdirenç [Ω·m]
        m_3d:     np.ndarray,    # (nx, ny, nz) chargeability [0..1]
        tau_3d:   np.ndarray,    # (nx, ny, nz) relaksasyon süresi [s]
        c_3d:     np.ndarray,    # (nx, ny, nz) frekans üsteli [0..1]
        x_coords: np.ndarray,    # (nx,) x koordinatları [m]
        z_coords: np.ndarray,    # (nz,) z koordinatları (derinlik) [m]
        obs_x:    np.ndarray,    # (n_stations,) yüzey istasyon konumları [m]
        frequencies: np.ndarray, # (n_freq,) ölçüm frekansları [Hz]
        a_spacing: float = 20.0, # Elektrot aralığı [m]
        n_max:     int   = 6,    # Maksimum dipole-dipole n katsayısı
    ) -> Dict[str, np.ndarray]:
        """
        3D Cole-Cole modelinden dipole-dipole IP pseudosection üretir.

        Returns: dict
            rho_a_dc     : DC görünür özdirenç pseudosection (n_n, n_stations) [Ω·m]
            chargeability: Görünür chargeability pseudosection (n_n, n_stations) [ms→adim]
            phase_mrad   : Faz pseudosection (n_n, n_stations, n_freq) [mrad]
            pseudo_x     : Pseudosection x koordinatları
            pseudo_z     : Pseudosection z koordinatları (negatif derinlik)
            K_factors    : Geometrik faktörler (n_n,)
        """
        nx, ny, nz = rho0_3d.shape

        # Tensörlere dönüştür
        rho0_t = torch.tensor(rho0_3d.reshape(-1), dtype=torch.float64, device=self.device)
        m_t    = torch.tensor(m_3d.reshape(-1),    dtype=torch.float64, device=self.device)
        tau_t  = torch.tensor(tau_3d.reshape(-1),  dtype=torch.float64, device=self.device)
        c_t    = torch.tensor(c_3d.reshape(-1),    dtype=torch.float64, device=self.device)

        omega = torch.tensor(
            2 * np.pi * frequencies, dtype=torch.float64, device=self.device
        )

        # Cole-Cole kompleks özdirenç: (N_cells, N_freq)
        rho_complex = self.cole_cole(omega, rho0_t, m_t, tau_t, c_t)

        # DC özdirenç (ω→0 limiti): ρ*(0) = ρ₀·(1-m)
        rho_dc = (rho0_t * (1 - m_t)).reshape(nx, ny, nz)

        # Chargeability (integral faz): m_apparent = (ρ_dc - ρ_inf) / ρ_dc
        # ρ_inf = ρ₀ (ω→∞ limiti)
        rho_inf = rho0_t.reshape(nx, ny, nz)
        m_apparent_3d = (rho_dc - rho_inf).abs() / rho_dc.clamp(min=1e-6)

        n_stations = len(obs_x)
        h_layers = torch.tensor(
            np.diff(np.abs(z_coords)), dtype=torch.float64, device=self.device
        )

        # Her n-faktörü için pseudosection satırı
        rho_a_pseudo  = np.zeros((n_max, n_stations))
        charge_pseudo = np.zeros((n_max, n_stations))
        phase_pseudo  = np.zeros((n_max, n_stations, len(frequencies)))
        K_factors     = np.zeros(n_max)
        pseudo_z      = np.zeros(n_max)

        for n_idx, n in enumerate(range(1, n_max + 1)):
            K = self.dipole_dipole_K(a_spacing, n)
            K_factors[n_idx] = K
            pseudo_z[n_idx]  = -0.519 * n * a_spacing  # Negatif derinlik

            for si, sx in enumerate(obs_x):
                # İstasyona en yakın x indeksini bul
                xi = int(np.argmin(np.abs(x_coords - sx)))
                xi = np.clip(xi, 0, nx - 1)

                # Y boyunca ortalama al (2D profil varsayımı)
                rho0_col  = rho0_t.reshape(nx, ny, nz)[xi, :, :].mean(dim=0)  # (nz,)
                m_col     = m_t.reshape(nx, ny, nz)[xi, :, :].mean(dim=0)
                tau_col   = tau_t.reshape(nx, ny, nz)[xi, :, :].mean(dim=0)
                c_col     = c_t.reshape(nx, ny, nz)[xi, :, :].mean(dim=0)

                rho_cc_col = self.cole_cole(omega, rho0_col, m_col, tau_col, c_col)
                rho_a, phase = self.apparent_resistivity_1d(
                    rho_cc_col, h_layers, a_spacing, n
                )

                rho_a_pseudo[n_idx, si]    = rho_a.mean().item()
                phase_pseudo[n_idx, si, :] = phase.cpu().numpy()

                # Görünür chargeability (Seigel, 1959)
                rho_dc_val  = (rho0_col * (1 - m_col)).mean().item()
                rho_inf_val = rho0_col.mean().item()
                charge_pseudo[n_idx, si] = max(0.0,
                    (rho_inf_val - rho_dc_val) / rho_inf_val
                    if rho_inf_val > 1e-6 else 0.0
                )

        pseudo_x = obs_x.copy()

        return {
            "rho_a_dc":      rho_a_pseudo,
            "chargeability": charge_pseudo,
            "phase_mrad":    phase_pseudo,
            "pseudo_x":      pseudo_x,
            "pseudo_z":      pseudo_z,
            "K_factors":     K_factors,
            "n_values":      np.arange(1, n_max + 1),
            "frequencies_hz": frequencies,
            "a_spacing_m":   a_spacing,
            "stats": {
                "rho_a_min_ohmm":  float(rho_a_pseudo.min()),
                "rho_a_max_ohmm":  float(rho_a_pseudo.max()),
                "rho_a_mean_ohmm": float(rho_a_pseudo.mean()),
                "charge_max":      float(charge_pseudo.max()),
                "charge_mean":     float(charge_pseudo.mean()),
                "phase_max_mrad":  float(np.abs(phase_pseudo).max()),
                "n_max":           n_max,
                "n_stations":      n_stations,
                "interpretation":  _interpret_ip(float(charge_pseudo.max())),
            }
        }


# ── Petrofizik: model → IP parametreleri ────────────────────────────────────
def model_to_ip_params(
    normalized_model: np.ndarray,
    # Host kaya (kireçtaşı — Beylikova)
    rho0_host:  float = 500.0,   # Ω·m
    m_host:     float = 0.02,    # adimensional
    tau_host:   float = 0.10,    # s
    c_host:     float = 0.50,    # adimensional
    # Cevher/alterasyon (sülfürlü REE zonu)
    rho0_ore:   float = 15.0,    # Ω·m  (Pelton et al. 1978, Tablo 1)
    m_ore:      float = 0.55,    # adimensional
    tau_ore:    float = 0.80,    # s
    c_ore:      float = 0.35,    # adimensional
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize [0,1] model → Cole-Cole IP parametre gridleri.

    Lineer interpolasyon: param(x) = param_host + f(x)·(param_ore - param_host)

    Referans: Pelton et al. (1978), Tablo 1 — Sülfürlü mineraller için
    tipik Cole-Cole parametreleri:
        Pirit:        m=0.1-0.8, τ=0.01-1s, c=0.1-0.6
        Kalkopirit:   m=0.2-0.6, τ=0.1-10s, c=0.2-0.5
        Pirotin:      m=0.1-0.5, τ=0.1-1s,  c=0.3-0.6
    """
    f = np.asarray(normalized_model, dtype=np.float64)
    rho0 = rho0_host + f * (rho0_ore - rho0_host)
    m    = m_host    + f * (m_ore    - m_host)
    tau  = tau_host  + f * (tau_ore  - tau_host)
    c    = c_host    + f * (c_ore    - c_host)
    return rho0, m, tau, c


def _interpret_ip(charge_max: float) -> str:
    """Görünür chargeability değerini jeolojik açıdan yorumlar."""
    if charge_max < 0.05:
        return "Düşük IP (<0.05) — sülfür minerali yok, sediman veya taze kaya"
    elif charge_max < 0.15:
        return "Orta IP (0.05-0.15) — dağınık sülfür mineralizasyonu, alterasyon"
    elif charge_max < 0.35:
        return "Yüksek IP (0.15-0.35) — yoğun sülfür, pirit/kalkopirit içeren alterasyon"
    else:
        return "Çok yüksek IP (>0.35) — masif sülfür veya grafit — güçlü cevher hedefi"


# ── Ana motor sınıfı (server.py entegrasyonu) ────────────────────────────────
class IPForwardMotor:
    """
    Server.py'e entegrasyon için ana sınıf.
    CSAMT1DForward ile aynı interface pattern'i.
    """

    def __init__(self):
        self.engine = ColeColeIP()

    def calculate(
        self,
        normalized_model: np.ndarray,
        x_coords:   np.ndarray,
        z_coords:   np.ndarray,
        obs_x:      np.ndarray,
        frequencies: np.ndarray = np.array([0.125, 0.5, 2.0, 8.0, 32.0]),
        a_spacing:  float = 20.0,
        n_max:      int   = 6,
        # Petrofizik parametreler
        rho0_host:  float = 500.0,
        m_host:     float = 0.02,
        tau_host:   float = 0.10,
        c_host:     float = 0.50,
        rho0_ore:   float = 15.0,
        m_ore:      float = 0.55,
        tau_ore:    float = 0.80,
        c_ore:      float = 0.35,
    ) -> Dict:
        """Model → IP pseudosection. server.py /api/ip/forward endpoint'i için."""

        rho0, m, tau, c = model_to_ip_params(
            normalized_model,
            rho0_host, m_host, tau_host, c_host,
            rho0_ore,  m_ore,  tau_ore,  c_ore,
        )

        result = self.engine.forward_pseudosection(
            rho0, m, tau, c,
            x_coords, z_coords, obs_x,
            frequencies, a_spacing, n_max,
        )

        return result


if __name__ == "__main__":
    print("[TEST] IP Forward Motoru — Cole-Cole Dipole-Dipole")

    import numpy as np

    # Sentetik Beylikova modeli
    nx, ny, nz = 16, 16, 16
    domain = 480.0
    dh = domain / nx

    rng = np.random.default_rng(42)
    model = np.zeros((nx, ny, nz))
    cx, cy, cz = nx//2, ny//2, nz//4
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                r = np.sqrt(((ix-cx)/4)**2 + ((iy-cy)/4)**2 + ((iz-cz)/3)**2)
                if r < 1.0:
                    model[ix, iy, iz] = max(model[ix, iy, iz], float(1.0 - r**0.7))

    x_c = np.linspace(dh/2, domain-dh/2, nx)
    z_c = np.linspace(dh/2, domain-dh/2, nz)
    obs_x = np.linspace(0, domain, 21)
    freqs = np.array([0.125, 0.5, 2.0, 8.0, 32.0])

    motor = IPForwardMotor()
    result = motor.calculate(model, x_c, z_c, obs_x, freqs)

    s = result["stats"]
    print(f"  Görünür özdirenç: {s['rho_a_min_ohmm']:.1f} — {s['rho_a_max_ohmm']:.1f} Ω·m")
    print(f"  Chargeability maks: {s['charge_max']:.3f}")
    print(f"  Faz maks: {s['phase_max_mrad']:.1f} mrad")
    print(f"  Pseudosection: {result['rho_a_dc'].shape}")
    print(f"  Yorum: {s['interpretation']}")
    print("[TEST] Tamamlandı.")

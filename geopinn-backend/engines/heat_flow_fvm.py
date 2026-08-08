"""
heat_flow_fvm.py — FVM tabanlı Isı Akışı (Heat Flow) İleri Yönlü Modelleme
GeoPINN Studio 3.0

Fizik:
    Durağan hal ısı iletimi (steady-state heat conduction):
        ∇·(k(x) ∇T) = -Q(x)

    Düzenli grid ve sabit k varsayımıyla sadeleşir:
        ∇²T = -Q(x) / k

    Q(x) = radyojenik ısı üretimi [W/m³]:
        Q = ρ · (c_U·A_U + c_Th·A_Th + c_K·A_K)

    Sınır koşulları (Beylikova için):
        Üst (z=0):  T = T_yüzey (Dirichlet, ~15°C)
        Alt (z=H):  T = T_taban (Dirichlet, ~60-80°C jeoisoterm)
        Yanlar:     ∂T/∂n = 0 (Neumann — ısı sızıntısı yok)

Referanslar:
    Birch (1954) — radyojenik ısı üretimi katsayıları
    Pollack & Chapman (1977) — jeotermik akı standartları
    Turcotte & Schubert (2002) — Geodynamics, Bölüm 4

Beylikova REE-F-Ba-Th için tipik değerler:
    U:  5-15 ppm   (monazit, xenotim içinde yoğunlaşır)
    Th: 20-80 ppm  (monazit baskın taşıyıcı)
    K:  1-4%       (K-feldispat, serizit alterasyon

    → Bu değerler "normal" granitik kayadan 3-5x yüksek ısı üretimi verir
    → Hidrotermal sistem + yüksek Q → SP anomalisiyle örtüşmeli
"""

import numpy as np
from typing import Optional, Tuple
try:
    from fvm_core import (
        build_padded_grid, embed_model,
        assemble_poisson_7point, apply_dirichlet_boundary,
        solve_poisson_field,
        vertical_gradient_at_surface,
    )
except ImportError:
    from engines.fvm_core import (
        build_padded_grid, embed_model,
        assemble_poisson_7point, apply_dirichlet_boundary,
        solve_poisson_field,
        vertical_gradient_at_surface,
    )
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── Radyojenik ısı üretimi katsayıları (Birch 1954, SI birimler) ──────────────
# [W / (kg · ppm)] veya [W / (kg · %)]
A_U  = 9.52e-5   # W kg⁻¹ ppm⁻¹  (U-238 + U-235 zinciri)
A_Th = 2.56e-5   # W kg⁻¹ ppm⁻¹  (Th-232 zinciri)
A_K  = 3.48e-6   # W kg⁻¹ %⁻¹    (K-40)

# Termal iletkenlik [W m⁻¹ K⁻¹]
K_GRANITE    = 3.0   # sağlam granit
K_CARBONATE  = 2.5   # kireçtaşı (Beylikova'nın host kayası)
K_ALTERED    = 1.8   # alterasyon zonu (kuvars-serizit ± kalsit)
K_SOIL       = 1.2   # toprak/dolgu


def radiogenic_heat_production(
    u_ppm: np.ndarray,
    th_ppm: np.ndarray,
    k_pct: np.ndarray,
    density_kg_m3: float = 2700.0,
) -> np.ndarray:
    """
    U, Th, K konsantrasyonlarından radyojenik ısı üretimi hesaplar.

    Parametreler:
        u_ppm        : Uranyum konsantrasyonu (ppm) — 3D grid
        th_ppm       : Toryum konsantrasyonu (ppm) — 3D grid
        k_pct        : Potasyum konsantrasyonu (%) — 3D grid
        density_kg_m3: Kaya yoğunluğu (kg/m³), varsayılan granit

    Döndürür:
        Q : Radyojenik ısı üretimi [W/m³] — 3D grid
    """
    return density_kg_m3 * (A_U * u_ppm + A_Th * th_ppm + A_K * k_pct)


def solve_heat_conduction(
    Q_padded: np.ndarray,
    k_thermal: float,
    dx: float, dy: float, dz: float,
    T_surface: float = 15.0,   # °C
    T_base: float    = 65.0,   # °C (yaklaşık 30°C/km jeotermik gradyan × ~1.5km)
) -> np.ndarray:
    """
    Durağan hal ısı iletimi: ∇²T = -Q/k

    Üst yüzeyde T = T_surface (Dirichlet),
    Alt yüzeyde  T = T_base   (Dirichlet),
    Yanlar: Neumann (sıfır akı) — zaten assemble_poisson_7point'te
    kenar hücreleri sıfır yapar, ama biz yanları Neumann bırakmak için
    sadece z=0 (üst) ve z=nz-1 (alt) sınırlarına Dirichlet uygularız.

    Döndürür:
        T : Sıcaklık alanı [°C] — (nx, ny, nz) grid
    """
    nx, ny, nz = Q_padded.shape
    N = nx * ny * nz

    # ∇²T = -Q/k  →  rhs = -Q/k
    rhs_field = -Q_padded / k_thermal

    # Laplacian matrisini kur
    A = assemble_poisson_7point(nx, ny, nz, dx, dy, dz)
    rhs = rhs_field.ravel().copy()

    # Sınır koşulları:
    # Sadece üst (iz=0) ve alt (iz=nz-1) Dirichlet — yanlar dokunulmaz
    idx3d = np.arange(N).reshape(nx, ny, nz)
    top_idx    = idx3d[:, :, 0].ravel()
    bottom_idx = idx3d[:, :, nz-1].ravel()

    # Üst yüzey: T = T_surface
    # Alt yüzey: T = T_base
    boundary_idx = np.concatenate([top_idx, bottom_idx])
    T_bc = np.concatenate([
        np.full(len(top_idx),    T_surface),
        np.full(len(bottom_idx), T_base),
    ])

    # Dirichlet uygulama: satırı sıfırla, diyagonale 1, RHS'e BC değeri yaz
    b_mask = np.zeros(N, dtype=bool)
    b_mask[boundary_idx] = True

    A_lil = A.tolil()
    A_lil[boundary_idx, :] = 0
    A_lil[boundary_idx, boundary_idx] = 1.0
    A_csr = A_lil.tocsr()

    rhs[boundary_idx] = T_bc
    # RHS düzeltmesi: iç noktalarda BC katkısını çıkar
    rhs_correction = A.dot(np.zeros(N))  # placeholder
    # Standart yaklaşım: A_modified @ T = rhs_modified
    # (LIL dönüşümü zaten satır sildiği için RHS doğrudan doğru)

    if N <= 60_000:
        T_flat = spla.spsolve(A_csr.tocsc(), rhs)
    else:
        diag = A_csr.diagonal()
        diag[diag == 0] = 1.0
        M = spla.LinearOperator(A_csr.shape, matvec=lambda v: v / diag)
        T_flat, info = spla.cg(A_csr, rhs, M=M, rtol=1e-8, maxiter=3000)
        if info != 0:
            raise RuntimeError(f"Isı iletimi CG çözücü yakınsamadı (info={info})")

    return T_flat.reshape(nx, ny, nz)


def surface_heat_flux(
    T: np.ndarray,
    k_thermal: float,
    pad_cells: int,
    dz: float,
) -> np.ndarray:
    """
    Yüzey ısı akışı: q_z = -k · ∂T/∂z  [W/m² → mW/m²]

    Sözleşme: aşağı pozitif derinlik → yüzeyde yukarı ısı akışı pozitif.
    Döndürür: (nx, ny) yüzey ısı akışı haritası [mW/m²]
    """
    # Yüzey: dolgu/iç grid arayüzü (pad_cells ve pad_cells-1 indeksleri arası)
    dTdz = (T[:, :, pad_cells] - T[:, :, pad_cells - 1]) / dz
    q_surface = -k_thermal * dTdz  # W/m²
    return q_surface * 1000.0  # mW/m²


def temperature_at_depth(
    T: np.ndarray,
    pad_cells: int,
    depth_idx: int,
) -> np.ndarray:
    """
    İstenen derinlikteki sıcaklık kesitini döndürür.
    depth_idx: iç grid'deki derinlik indeksi (0=yüzey, n-1=taban)
    """
    return T[:, :, pad_cells + depth_idx]


class HeatFlowFVM:
    """
    Isı akışı FVM motoru — server.py'e entegrasyon için.

    Kullanım (server.py'de):
        eng = HeatFlowFVM(k_thermal=2.5)
        result = eng.calculate(
            u_ppm_grid, th_ppm_grid, k_pct_grid,
            x_coords, y_coords, z_coords
        )
    """

    def __init__(
        self,
        k_thermal: float   = 2.5,     # W m⁻¹ K⁻¹ (kireçtaşı host)
        density:   float   = 2700.0,  # kg/m³
        T_surface: float   = 15.0,    # °C
        T_base:    float   = 65.0,    # °C
        pad_cells: int     = 8,
    ):
        self.k         = k_thermal
        self.density   = density
        self.T_surface = T_surface
        self.T_base    = T_base
        self.pad_cells = pad_cells

    def calculate(
        self,
        u_ppm:   np.ndarray,
        th_ppm:  np.ndarray,
        k_pct:   np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray,
        depth_indices: Optional[list] = None,
    ) -> dict:
        """
        U/Th/K konsantrasyonlarından yüzey ısı akışı ve sıcaklık alanı hesaplar.

        Parametreler:
            u_ppm, th_ppm, k_pct: (nx, ny, nz) konsantrasyon gridleri
            x/y/z_coords: grid koordinatları (metre)
            depth_indices: sıcaklık kesiti alınacak iz indeksleri (None=yok)

        Döndürür: dict
            heat_flux_mw_m2  : (nx, ny) yüzey ısı akışı haritası [mW/m²]
            T_field          : (nx, ny, nz) sıcaklık alanı [°C]
            Q_field          : (nx, ny, nz) radyojenik ısı üretimi [W/m³]
            T_at_depth       : {depth_idx: (nx,ny) array} — istenen kesitler
            stats            : özet istatistikler
        """
        u_ppm  = np.asarray(u_ppm,  dtype=np.float64)
        th_ppm = np.asarray(th_ppm, dtype=np.float64)
        k_pct  = np.asarray(k_pct,  dtype=np.float64)

        # Grid hazırla
        x_p, y_p, z_p, dx, dy, dz = build_padded_grid(
            x_coords, y_coords, z_coords, pad_cells=self.pad_cells
        )

        # Radyojenik ısı üretimi
        Q_inner = radiogenic_heat_production(u_ppm, th_ppm, k_pct, self.density)
        Q_padded = embed_model(Q_inner, self.pad_cells)

        # Isı iletimi çöz
        T = solve_heat_conduction(
            Q_padded, self.k, dx, dy, dz,
            T_surface=self.T_surface, T_base=self.T_base,
        )

        # Yüzey ısı akışı
        q_surf = surface_heat_flux(T, self.k, self.pad_cells, dz)

        # İstenen derinlik kesitleri
        T_at_depth = {}
        if depth_indices:
            for di in depth_indices:
                T_at_depth[int(di)] = temperature_at_depth(T, self.pad_cells, di)

        # İç grid sıcaklık alanı (dolgu olmadan)
        p = self.pad_cells
        nx_i, ny_i, nz_i = u_ppm.shape
        T_inner = T[p:p+nx_i, p:p+ny_i, p:p+nz_i]

        # İstatistikler
        q_vals = q_surf[p:p+nx_i, p:p+ny_i]  # dolgu olmadan
        stats = {
            "heat_flux_mean_mw_m2":  float(np.mean(q_vals)),
            "heat_flux_max_mw_m2":   float(np.max(q_vals)),
            "heat_flux_min_mw_m2":   float(np.min(q_vals)),
            "heat_flux_std_mw_m2":   float(np.std(q_vals)),
            "Q_mean_uw_m3":          float(np.mean(Q_inner) * 1e6),   # μW/m³
            "Q_max_uw_m3":           float(np.max(Q_inner) * 1e6),
            "T_max_inner_c":         float(np.max(T_inner)),
            "T_at_500m_c":           float(np.mean(T_inner[:, :, min(nz_i//3, nz_i-1)])),
            "k_thermal_w_mk":        self.k,
            "interpretation": _interpret_heat_flux(float(np.mean(q_vals))),
        }

        return {
            "heat_flux_mw_m2": q_vals.tolist(),
            "T_field":         T_inner.tolist(),
            "Q_field":         Q_inner.tolist(),
            "T_at_depth":      {str(k): v.tolist() for k, v in T_at_depth.items()},
            "stats":           stats,
        }


def _interpret_heat_flux(q_mean: float) -> str:
    """Yüzey ısı akışı değerini jeolojik açıdan yorumlar."""
    if q_mean < 40:
        return "Düşük (<40 mW/m²) — kraton kalkanı, soğuk litosfer"
    elif q_mean < 60:
        return "Normal (40-60 mW/m²) — tipik kıtasal kabuk"
    elif q_mean < 90:
        return "Yüksek (60-90 mW/m²) — radyojenik zengin granit veya genç tektonik"
    elif q_mean < 150:
        return "Çok yüksek (90-150 mW/m²) — aktif hidrotermal sistem veya volkanik"
    else:
        return "Anormal (>150 mW/m²) — aktif hidrotermal/magmatik kaynak, REE yatağı olası"


# ── Radyometri verisi okuma yardımcıları ─────────────────────────────────────
def load_radiometry_csv(path: str) -> dict:
    """
    Saha radyometri verisi okur.

    Beklenen sütunlar (esnek sıra):
        x, y, [z], U_ppm, Th_ppm, K_pct, [TC_cps], [dose_nGy_h]

    Desteklenen ayraçlar: virgül, tab, boşluk
    Döndürür: dict of numpy arrays
    """
    import csv
    with open(path, newline='', encoding='utf-8-sig') as f:
        sample = f.read(2048)
    # Ayraç tespiti
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=',\t; ')
        sep = dialect.delimiter
    except Exception:
        sep = ','

    data = np.genfromtxt(path, delimiter=sep, names=True, dtype=None,
                          encoding='utf-8-sig', invalid_raise=False)

    result = {}
    col_map = {
        'x': ['x','X','easting','lon','longitude','E'],
        'y': ['y','Y','northing','lat','latitude','N'],
        'z': ['z','Z','elev','elevation','alt','altitude'],
        'U': ['u_ppm','U_ppm','U','uranium','u','U_PPM'],
        'Th': ['th_ppm','Th_ppm','Th','thorium','th','TH_PPM','Th_PPM'],
        'K': ['k_pct','K_pct','K','potassium','k','K_PCT','K_pct'],
        'TC': ['tc_cps','TC_cps','TC','total_count','cps','CPS','total_cps'],
        'dose': ['dose','dose_ngy_h','dose_rate','ngy_h'],
    }
    names_lower = [n.lower() for n in data.dtype.names]
    for key, candidates in col_map.items():
        for cand in candidates:
            if cand.lower() in names_lower:
                idx = names_lower.index(cand.lower())
                col_name = data.dtype.names[idx]
                result[key] = data[col_name].astype(np.float64)
                break

    if 'x' not in result or 'y' not in result:
        raise ValueError("CSV'de x/y koordinat sütunları bulunamadı.")

    return result


def interpolate_to_grid(
    obs_x: np.ndarray, obs_y: np.ndarray, obs_values: np.ndarray,
    grid_x: np.ndarray, grid_y: np.ndarray,
    method: str = 'linear',
) -> np.ndarray:
    """
    Düzensiz saha ölçümlerini düzenli grid'e enterpolasyon yapar.
    method: 'linear', 'cubic', 'nearest'
    """
    from scipy.interpolate import griddata
    pts  = np.column_stack([obs_x, obs_y])
    xi, yi = np.meshgrid(grid_x, grid_y, indexing='ij')
    grid_pts = np.column_stack([xi.ravel(), yi.ravel()])
    zi = griddata(pts, obs_values, grid_pts, method=method, fill_value=float(np.nanmedian(obs_values)))
    return zi.reshape(len(grid_x), len(grid_y))


if __name__ == "__main__":
    print("[TEST] HeatFlowFVM motoru test ediliyor...")

    # Sentetik Beylikova benzeri model
    nx, ny, nz = 16, 16, 16
    domain = 480.0
    dh = domain / nx

    x_c = np.linspace(dh/2, domain-dh/2, nx)
    y_c = np.linspace(dh/2, domain-dh/2, ny)
    z_c = np.linspace(dh/2, domain-dh/2, nz)

    # Arka plan + cevher bölgesi
    u_bg,  th_bg,  k_bg  = 3.0, 12.0, 2.5    # ppm, ppm, %
    u_ore, th_ore, k_ore = 12.0, 55.0, 4.2   # Beylikova anomali

    u  = np.full((nx, ny, nz), u_bg)
    th = np.full((nx, ny, nz), th_bg)
    k  = np.full((nx, ny, nz), k_bg)

    # Cevher gövdesi (merkez üst)
    cx, cy, cz = nx//2, ny//2, nz//4
    for i in range(nx):
        for j in range(ny):
            for l in range(nz):
                r = np.sqrt(((i-cx)/5)**2 + ((j-cy)/5)**2 + ((l-cz)/3)**2)
                if r < 1.0:
                    u[i,j,l]  = u_ore
                    th[i,j,l] = th_ore
                    k[i,j,l]  = k_ore

    eng = HeatFlowFVM(k_thermal=2.5, T_surface=15.0, T_base=65.0)
    result = eng.calculate(u, th, k, x_c, y_c, z_c, depth_indices=[0, 4, 8])

    print(f"  Yüzey ısı akışı: {result['stats']['heat_flux_mean_mw_m2']:.1f} mW/m²")
    print(f"  Maks. ısı akışı: {result['stats']['heat_flux_max_mw_m2']:.1f} mW/m²")
    print(f"  Radyojenik Q:    {result['stats']['Q_max_uw_m3']:.2f} μW/m³")
    print(f"  Yorum: {result['stats']['interpretation']}")
    print("[TEST] Tamamlandı.")

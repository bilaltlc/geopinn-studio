"""
engines/harmonica_validation.py

Fatiando a Terra / Harmonica entegrasyonu.

İki amaç:
  1) DOĞRULAMA: mevcut gravity_prism.PrismGravityForward ve
     magnetic_prism.PrismMagneticForward çıktılarını Harmonica'nın
     referans implementasyonuyla karşılaştırır (regresyon testi).
  2) SENTETİK VERİ ÜRETİMİ: eğitim/test için Harmonica ile hızlı,
     bilinen geometrili sentetik anomali haritaları üretir.

Kurulum (backend ortamında bir kez çalıştır):
    pip install harmonica verde

Kullanım (server.py veya doğrudan):
    from engines.harmonica_validation import validate_gravity, generate_synthetic_anomaly
    report = validate_gravity(density_grid, x_coords, y_coords, z_coords, obs_x, obs_y)
    print(report)
"""

import numpy as np
import warnings

try:
    import harmonica as hm
    HARMONICA_OK = True
except ImportError:
    HARMONICA_OK = False
    warnings.warn(
        "Harmonica kurulu değil. 'pip install harmonica verde' ile kurun. "
        "Doğrulama fonksiyonları devre dışı.",
        ImportWarning,
        stacklevel=2,
    )

G_CONST = 6.6743e-11
SI_TO_MGAL = 1e5
MU_0 = 4 * np.pi * 1e-7


def _require_harmonica():
    if not HARMONICA_OK:
        raise ImportError(
            "Harmonica kurulu değil. Lütfen 'pip install harmonica verde' çalıştırın."
        )


# ── Yardımcı: numpy density grid -> harmonica prism listesi ──────────────────

def _grid_to_prisms(density_or_chi, x_coords, y_coords, z_coords, threshold=1e-6):
    """3D voksel gridinı (nx,ny,nz) Harmonica prism dizisine çevirir.

    Sıfır (veya threshold altı) vokseller atlanır — büyük grids için bellek tasarrufu.
    Döndürür: prisms (N,6) float64 array [x1,x2,y1,y2,z1_derinlik,z2_derinlik],
              prop   (N,)  float64 array (yoğunluk veya duyarlılık)

    Koordinat sözleşmesi: Harmonica prism z ekseni YUKARI-POZİTİF (yüzey z=0,
    derinlik negatif). gravity_prism.py AŞAĞI-POZİTİF kullanıyor — burada
    dönüştürülüyor.
    """
    arr = np.asarray(density_or_chi, dtype=np.float64)
    nx, ny, nz = arr.shape
    dx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else float(x_coords[0])
    dy = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else float(y_coords[0])
    dz = float(z_coords[1] - z_coords[0]) if len(z_coords) > 1 else float(z_coords[0])

    prisms_list, prop_list = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                val = arr[i, j, k]
                if abs(val) < threshold:
                    continue
                x1 = float(x_coords[i]) - dx / 2
                x2 = float(x_coords[i]) + dx / 2
                y1 = float(y_coords[j]) - dy / 2
                y2 = float(y_coords[j]) + dy / 2
                # Derinlik dönüşümü: aşağı-pozitif z -> yukari-pozitif z (negatif)
                z_top = -float(z_coords[k]) + dz / 2   # yüzeye daha yakın = daha az negatif
                z_bot = -float(z_coords[k]) - dz / 2
                prisms_list.append([x1, x2, y1, y2, z_bot, z_top])
                prop_list.append(val)

    if not prisms_list:
        return np.zeros((0, 6)), np.zeros(0)
    return np.array(prisms_list, dtype=np.float64), np.array(prop_list, dtype=np.float64)


# ── 1) Gravite Doğrulama ─────────────────────────────────────────────────────

def validate_gravity(density_contrast_kg_m3, x_coords, y_coords, z_coords,
                     obs_x, obs_y, obs_z=0.0, our_gz=None):
    """Kendi Nagy-kernel implementasyonumuzu Harmonica referansıyla karşılaştırır.

    density_contrast_kg_m3: (nx,ny,nz) numpy array, kg/m³ kontrast
    x/y/z_coords: grid koordinatları (metre)
    obs_x, obs_y: gözlem noktaları (meshgrid veya 1D)
    obs_z: gözlem yüksekliği (m, varsayılan 0)
    our_gz: (opsiyonel) kendi motorumuzun çıktısı (mGal); verilmezse içeride
            hesaplanır (gravity_prism modülü gerekir)

    Döndürür: dict — harmonica_gz, our_gz, max_diff_mgal, rmse_mgal, rel_rmse_pct
    """
    _require_harmonica()

    density = np.asarray(density_contrast_kg_m3, dtype=np.float64)
    obs_x_f = np.asarray(obs_x, dtype=np.float64).ravel()
    obs_y_f = np.asarray(obs_y, dtype=np.float64).ravel()
    obs_z_f = np.full_like(obs_x_f, float(obs_z))

    prisms, dens = _grid_to_prisms(density, x_coords, y_coords, z_coords)

    if prisms.shape[0] == 0:
        return {"error": "Sıfır-dışı voksel bulunamadı, eşiği düşürün."}

    # Harmonica gravite (g_z, SI birimi m/s²) -> mGal
    gz_hm_si = hm.prism_gravity(
        (obs_x_f, obs_y_f, obs_z_f),
        prisms,
        dens,
        field="g_z",
    )
    gz_hm_mgal = gz_hm_si * SI_TO_MGAL

    # Kendi motorumuz
    if our_gz is None:
        try:
            from engines.gravity_prism import PrismGravityForward
            eng = PrismGravityForward()
            gz_our = eng.calculate(
                density, x_coords, y_coords, z_coords,
                obs_x, obs_y, obs_z=float(obs_z),
                track_gradients=False,
            )
            if hasattr(gz_our, "cpu"):
                gz_our = gz_our.cpu().numpy()
            our_gz = gz_our.ravel()
        except Exception as e:
            our_gz = None
            warnings.warn(f"Kendi motor hesabı başarısız: {e}")

    result = {"harmonica_gz_mgal": gz_hm_mgal.reshape(obs_x.shape if hasattr(obs_x, 'shape') else (-1,))}

    if our_gz is not None:
        our_flat = np.asarray(our_gz, dtype=np.float64).ravel()
        diff = gz_hm_mgal - our_flat
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        rel = rmse / (float(np.std(gz_hm_mgal)) + 1e-12) * 100
        result.update({
            "our_gz_mgal": our_flat.reshape(result["harmonica_gz_mgal"].shape),
            "max_diff_mgal": float(np.max(np.abs(diff))),
            "rmse_mgal": rmse,
            "rel_rmse_pct": rel,
            "status": "OK" if rel < 5.0 else "UYARI: %5'ten büyük sapma",
        })

    return result


# ── 2) Manyetik Doğrulama ────────────────────────────────────────────────────

def validate_magnetic(chi_matrix, x_coords, y_coords, z_coords,
                      obs_x, obs_y, obs_z=0.0,
                      inc_deg=60.0, dec_deg=5.0, b0_nt=47000.0,
                      our_dt=None):
    """Kendi Bhattacharyya TMI implementasyonumuzu Harmonica'nın referansıyla karşılaştırır.

    Harmonica v0.7 doğrudan TMI hesaplamıyor (b_n/b_e/b_u bileşenler veriyor),
    bu yüzden bileşenlerden TMI projeksiyonu manuel yapılıyor.
    """
    _require_harmonica()

    chi = np.asarray(chi_matrix, dtype=np.float64)
    obs_x_f = np.asarray(obs_x, dtype=np.float64).ravel()
    obs_y_f = np.asarray(obs_y, dtype=np.float64).ravel()
    obs_z_f = np.full_like(obs_x_f, float(obs_z))

    inc = np.radians(inc_deg)
    dec = np.radians(dec_deg)
    L = np.cos(inc) * np.cos(dec)
    M = np.cos(inc) * np.sin(dec)
    N = np.sin(inc)
    H0 = b0_nt * 1e-9 / MU_0  # A/m

    prisms, chi_vals = _grid_to_prisms(chi, x_coords, y_coords, z_coords, threshold=1e-8)
    if prisms.shape[0] == 0:
        return {"error": "Sıfır-dışı duyarlılık vokseli bulunamadı."}

    # Manyetizasyon vektörü: M = chi * H0 * (L, M, N) yön vektörü
    magnetization = (
        chi_vals * H0 * L,
        chi_vals * H0 * M,
        chi_vals * H0 * N,
    )

    # Harmonica manyetik bileşenler (b_n=kuzey, b_e=doğu, b_u=yukarı), Tesla
    bn = hm.prism_magnetic((obs_x_f, obs_y_f, obs_z_f), prisms, magnetization, field="b_n")
    be = hm.prism_magnetic((obs_x_f, obs_y_f, obs_z_f), prisms, magnetization, field="b_e")
    bu = hm.prism_magnetic((obs_x_f, obs_y_f, obs_z_f), prisms, magnetization, field="b_u")

    # TMI projeksiyonu: dT = L*Bx + M*By + N*Bz (nT)
    # Harmonica: b_n=kuzey(x), b_e=doğu(y), b_u=yukarı(z=-aşağı)
    dt_hm_nt = (L * bn + M * be - N * bu) * 1e9  # Tesla -> nT, b_u yukarı=+z, N aşağı yön

    shape = obs_x.shape if hasattr(obs_x, 'shape') else (-1,)
    result = {"harmonica_tmi_nt": dt_hm_nt.reshape(shape)}

    if our_dt is not None:
        our_flat = np.asarray(our_dt, dtype=np.float64).ravel()
        diff = dt_hm_nt - our_flat
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        rel = rmse / (float(np.std(dt_hm_nt)) + 1e-12) * 100
        result.update({
            "our_tmi_nt": our_flat.reshape(shape),
            "max_diff_nt": float(np.max(np.abs(diff))),
            "rmse_nt": rmse,
            "rel_rmse_pct": rel,
            "status": "OK" if rel < 5.0 else "UYARI: %5'ten büyük sapma",
        })

    return result


# ── 3) Sentetik Anomali Üretimi ──────────────────────────────────────────────

def generate_synthetic_anomaly(
    nx=16, ny=16, nz=16,
    domain_m=480.0,
    n_bodies=2,
    density_host=2700.0,
    density_contrast=500.0,
    inc_deg=60.0, dec_deg=5.0, b0_nt=47000.0,
    chi_host=1e-4, chi_contrast=3e-4,
    seed=42,
):
    """Bilinen geometrili (rastgele dikdörtgen prizma) sentetik model üretir.

    Döndürür:
        geometry  : (nx,ny,nz) [0,1] normalize geometri alanı
        gz_mgal   : (21,21) Bouguer gravite anomalisi (mGal, Harmonica)
        tmi_nt    : (21,21) TMI manyetik anomalisi (nT, Harmonica)
        prism_params: prizma koordinatlarının listesi (doğrulama için)
    """
    _require_harmonica()
    rng = np.random.default_rng(seed)

    dh = domain_m / nx
    x_c = np.linspace(dh / 2, domain_m - dh / 2, nx)
    y_c = np.linspace(dh / 2, domain_m - dh / 2, ny)
    z_c = np.linspace(dh / 2, domain_m - dh / 2, nz)

    geometry = np.zeros((nx, ny, nz), dtype=np.float64)
    body_params = []

    for _ in range(n_bodies):
        xi = rng.integers(1, max(2, nx - 3))
        yi = rng.integers(1, max(2, ny - 3))
        zi = rng.integers(1, max(2, nz // 2))          # üst yarıya koy (daha gerçekçi)
        xw = rng.integers(2, max(3, nx // 3))
        yw = rng.integers(2, max(3, ny // 3))
        zw = rng.integers(1, max(2, nz // 4))
        xe = min(xi + xw, nx)
        ye = min(yi + yw, ny)
        ze = min(zi + zw, nz)
        geometry[xi:xe, yi:ye, zi:ze] = 1.0
        body_params.append(dict(xi=xi, yi=yi, zi=zi, xe=xe, ye=ye, ze=ze))

    density = density_host + density_contrast * geometry
    density_contrast_grid = density_contrast * geometry

    obs_1d = np.linspace(0, domain_m, 21)
    obs_x, obs_y = np.meshgrid(obs_1d, obs_1d)
    obs_z = np.zeros_like(obs_x)

    prisms_grav, dens_vals = _grid_to_prisms(density_contrast_grid, x_c, y_c, z_c)
    gz_mgal = np.zeros(obs_x.shape)
    if prisms_grav.shape[0] > 0:
        gz_si = hm.prism_gravity(
            (obs_x.ravel(), obs_y.ravel(), obs_z.ravel()),
            prisms_grav, dens_vals, field="g_z",
        )
        gz_mgal = (gz_si * SI_TO_MGAL).reshape(obs_x.shape)

    chi_grid = chi_contrast * geometry
    prisms_mag, chi_vals = _grid_to_prisms(chi_grid, x_c, y_c, z_c, threshold=1e-8)
    tmi_nt = np.zeros(obs_x.shape)
    if prisms_mag.shape[0] > 0:
        inc = np.radians(inc_deg)
        dec = np.radians(dec_deg)
        L, M, N = np.cos(inc) * np.cos(dec), np.cos(inc) * np.sin(dec), np.sin(inc)
        H0 = b0_nt * 1e-9 / MU_0
        magnetization = (
            chi_vals * H0 * L,   # easting
            chi_vals * H0 * M,   # northing
            chi_vals * H0 * N,   # upward
        )
        coords = (obs_x.ravel(), obs_y.ravel(), obs_z.ravel())
        bn = hm.prism_magnetic(coords, prisms_mag, magnetization, field="b_n")
        be = hm.prism_magnetic(coords, prisms_mag, magnetization, field="b_e")
        bu = hm.prism_magnetic(coords, prisms_mag, magnetization, field="b_u")
        tmi_nt = ((L * bn + M * be - N * bu) * 1e9).reshape(obs_x.shape)

    return {
        "geometry": geometry,
        "gz_mgal": gz_mgal,
        "tmi_nt": tmi_nt,
        "obs_x": obs_x,
        "obs_y": obs_y,
        "x_coords": x_c,
        "y_coords": y_c,
        "z_coords": z_c,
        "body_params": body_params,
        "domain_m": domain_m,
    }


# ── 4) Hızlı kendi kendini test ──────────────────────────────────────────────

if __name__ == "__main__":
    print("[TEST] Harmonica entegrasyon testi başlatılıyor...")

    result = generate_synthetic_anomaly(nx=16, ny=16, nz=16, n_bodies=2, seed=0)
    print(f"  Geometri şekli   : {result['geometry'].shape}")
    print(f"  Gravite (mGal)   : min={result['gz_mgal'].min():.3f}  max={result['gz_mgal'].max():.3f}")
    print(f"  TMI (nT)         : min={result['tmi_nt'].min():.3f}  max={result['tmi_nt'].max():.3f}")

    # Gravite doğrulaması (kendi motor olmasa bile sadece Harmonica çıktısı)
    geo = result["geometry"] * 500.0  # kontrast kg/m3
    val_report = validate_gravity(
        geo, result["x_coords"], result["y_coords"], result["z_coords"],
        result["obs_x"], result["obs_y"],
    )
    print(f"  Harmonica gz max : {val_report['harmonica_gz_mgal'].max():.3f} mGal")
    print("[TEST] Tamamlandı.")

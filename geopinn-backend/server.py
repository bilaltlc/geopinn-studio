"""
server.py  —  GeoPINN Studio 3.0 Backend
FastAPI + fizik motorları (engines/ klasöründe)

Çalıştırma (geliştirme):
    uvicorn server:app --reload --host 127.0.0.1 --port 8000

Gereksinimler:
    pip install fastapi uvicorn numpy scipy python-multipart

Not (Electron paketleme):
    Bu dosya PyInstaller ile tek dosya exe'ye çevrilip Electron'un
    "resources/backend/" klasörüne gömülüyor. Bkz. PAKETLEME_REHBERI.md
"""

import os
import re
import shutil
import numpy as np
import scipy.ndimage
import torch
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Fizik motorları (proje içindeki engines/ klasöründen)
from engines import gravity_prism, magnetic_prism, csamt_1d
try:
    from engines.pinn_stub import pinn_status as _pinn_status, infer as _pinn_infer
    PINN_AVAILABLE = True
except ImportError:
    PINN_AVAILABLE = False
PINN_STUB_OK = PINN_AVAILABLE  # geriye dönük uyumluluk

try:
    from engines.ip_forward import IPForwardMotor, model_to_ip_params
    IP_AVAILABLE = True
except ImportError:
    IP_AVAILABLE = False

try:
    from engines.sp_forward import SPForwardMotor
    SP_AVAILABLE = True
except ImportError:
    SP_AVAILABLE = False

try:
    from engines.data_fusion import DataFusionEngine
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False

try:
    from engines.seismic_refrac import SeismicRefracMotor, VP_CLASSES, classify_vp
    SEISMIC_AVAILABLE = True
except ImportError:
    SEISMIC_AVAILABLE = False

try:
    from engines.field_gridding import FieldGriddingMotor
    GRIDDING_AVAILABLE = True
except ImportError:
    GRIDDING_AVAILABLE = False

try:
    from engines.export_geospatial import GeoExportMotor
    EXPORT_GEO_AVAILABLE = True
except ImportError:
    EXPORT_GEO_AVAILABLE = False

try:
    from engines.gravity_fvm import PrismGravityForwardFVM
    from engines.magnetic_fvm import PrismMagneticForwardFVM
    FVM_AVAILABLE = True
except ImportError:
    FVM_AVAILABLE = False

try:
    from engines.gravity_prism import PrismGravityForward
    GRAV_AVAILABLE = True
except ImportError:
    GRAV_AVAILABLE = False

try:
    from engines.magnetic_prism import PrismMagneticForward
    MAG_AVAILABLE = True
except ImportError:
    MAG_AVAILABLE = False

try:
    from engines.heat_flow_fvm import HeatFlowFVM, radiogenic_heat_production
    from engines.radiometry import (
        forward_radiometry_surface, ree_alteration_index,
        radiometry_stats, read_field_data, detect_format, SUPPORTED_FORMATS,
    )
    RADIOMETRY_AVAILABLE = True
except ImportError:
    RADIOMETRY_AVAILABLE = False

app = FastAPI(title="GeoPINN Studio API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Startup: örnek veriyi uploads/'a kopyala ──────────────────────────────────
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
if os.path.exists(SAMPLE_DIR):
    for fname in os.listdir(SAMPLE_DIR):
        if fname.endswith(".npy"):
            src = os.path.join(SAMPLE_DIR, fname)
            dst = os.path.join(UPLOAD_DIR, fname)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
                print(f"[startup] Örnek veri kopyalandı: {fname}")

DEFAULT_DATASET = "Y_beylikova.npy"

# ── Beylikova domain parametreleri (data_factory.py ile TUTARLI) ──────────────
DOMAIN_EXTENT = 480.0     # m
NBC_NATIVE = 64           # Fabrika grid boyutu (varsayılan; yüklenen dosyaya göre değişebilir)
DH_NATIVE = DOMAIN_EXTENT / NBC_NATIVE   # 7.5 m

# Forward hesaplama için daha küçük, hızlı bir grid kullanıyoruz.
NBC_FORWARD = 16
DH_FORWARD = DOMAIN_EXTENT / NBC_FORWARD   # 30 m

# Petrofizik bağıntılar (run_physics ve joint_inversion arasında TUTARLI olmalı)
DENSITY_SCALE = 2000.0       # kg/m3  (Δρ = 2.0 g/cm3)
SUSCEPT_SCALE = 3e-4         # SI     (Δχ)
RES_HOST = 500.0             # ohm.m
RES_RATIO = 0.10             # ρ(x) = RES_HOST * RES_RATIO^m


# ── İstek modelleri ───────────────────────────────────────────────────────────
class SimulationRequest(BaseModel):
    grav_active: bool = True
    mag_active: bool = True
    csamt_active: bool = False
    selected_index: int = 0
    dataset: Optional[str] = None
    dataset_gravmag: Optional[str] = None
    dataset_csamt: Optional[str] = None
    engine_mode: str = "prism"   # "prism" (analitik Nagy/Bhattacharyya) | "fvm" (Poisson bounded)


class JointInversionRequest(BaseModel):
    grav_active: bool = True
    mag_active: bool = True
    csamt_active: bool = False
    selected_index: int = 0
    dataset: Optional[str] = None            # Y_... model küpü (None => demo/sentetik)
    dataset_grav_mag: Optional[str] = None   # X_mag_grav_... gerçek gözlem (None => motor kendi forward'lar)
    dataset_csamt: Optional[str] = None      # X_csamt_... gerçek gözlem (None => motor kendi forward'lar)
    n_iter: int = Field(default=60, ge=4, le=200)   # frontend max 200'e çıkarıldı
    weights: Dict[str, float] = Field(default_factory=lambda: {"grav": 1.0, "mag": 1.0, "csamt": 1.0})
    reg_lambda: float = 0.001   # reg_lambda tarama v4: RMSE knee=0.005, plateau λ≤0.002
    seed: int = 42
    nbc_forward: int = Field(default=NBC_FORWARD, ge=8, le=64)  # ters çözüm grid boyutu (16 hızlı, 64 native ile birebir)


# ── Yardımcılar: dosya güvenliği ─────────────────────────────────────────────
def _safe_npy_name(name: str) -> str:
    """Path traversal'a karşı korumalı, sadece .npy dosya adına izin ver."""
    base = os.path.basename(name)
    if not re.match(r"^[A-Za-z0-9_\-\.]+\.npy$", base):
        raise HTTPException(status_code=400, detail="Geçersiz dosya adı (yalnızca .npy).")
    return base


def _resolve_dataset_path(dataset: Optional[str]) -> Optional[str]:
    """dataset verilmişse uploads/ içinde ara; yoksa varsayılan dosyayı dene."""
    name = dataset or DEFAULT_DATASET
    safe = _safe_npy_name(name)
    path = os.path.join(UPLOAD_DIR, safe)
    return path if os.path.exists(path) else None


def _classify_dataset_shape(shape: list) -> str:
    """Yüklenen .npy dosyasının türünü şekline bakarak sınıflandırır.

    - 'model'        : (n, d, d, d) ya da (d, d, d) — native model küpü (Y_...)
    - 'obs_grav_mag' : (n, 21, 21, 2) — gerçek gravite+manyetik gözlem (X_mag_grav_...)
    - 'obs_csamt'     : (n, 21, 21, n_freq) — gerçek CSAMT gözlem (X_csamt_...)
    - 'unknown'       : tanınmayan şekil
    """
    if len(shape) == 4:
        n, a, b, c = shape
        if a == b == c:
            return "model"
        if a == b and c == 2:
            return "obs_grav_mag"
        if a == b and 1 <= c <= 12:
            return "obs_csamt"
    elif len(shape) == 3:
        a, b, c = shape
        if a == b == c:
            return "model"
    return "unknown"


def _load_indexed_npy(path: str, selected_index: int) -> np.ndarray:
    """(n, ...) ya da (...) şeklindeki bir .npy dosyasından tek bir örneği okur."""
    arr = np.load(path)
    if arr.ndim >= 1 and arr.shape[0] > 1 and arr.ndim in (4,):
        idx = selected_index % arr.shape[0]
        return arr[idx].astype(np.float64)
    return arr.astype(np.float64)


# ── Yardımcılar: model yükleme / grid hazırlama ──────────────────────────────
def load_model_native(dataset: Optional[str], selected_index: int):
    """Yüklü .npy dosyasından ya da sentetik demo cevherinden native-grid model üretir."""
    data_path = _resolve_dataset_path(dataset)

    if data_path is not None:
        raw = np.load(data_path)
        if raw.ndim == 4:
            idx = selected_index % raw.shape[0]
            model_native = raw[idx].astype(np.float64)
        elif raw.ndim == 3:
            model_native = raw.astype(np.float64)
        else:
            raise HTTPException(status_code=400, detail=f"Beklenmedik boyut: {raw.ndim}")
        used_path = data_path
    else:
        # Demo: Beylikova tipi sentetik cevher gövdesi
        model_native = np.zeros((NBC_NATIVE, NBC_NATIVE, NBC_NATIVE))
        cx, cy, cz = NBC_NATIVE // 2, NBC_NATIVE // 2, NBC_NATIVE // 3
        for i in range(NBC_NATIVE):
            for j in range(NBC_NATIVE):
                for k in range(NBC_NATIVE):
                    d = np.sqrt(((i - cx) / 8) ** 2 + ((j - cy) / 8) ** 2 + ((k - cz) / 4) ** 2)
                    model_native[i, j, k] = np.exp(-d ** 2) * 2.0
        used_path = None

    return model_native, used_path


def resample_to_forward(model_native: np.ndarray, nbc: int = NBC_FORWARD):
    """Native grid'i (jeolojik yapıyı koruyarak) forward grid'e indirger; koordinat/istasyon dizilerini döndürür.

    nbc: hedef grid boyutu. Varsayılan NBC_FORWARD (16, hızlı). Gerçek X_mag_grav/X_csamt
    verisiyle çalışırken çözünürlük uyuşmazlığını azaltmak için 32/64 seçilebilir
    (data_factory.py native grid'i zaten 64 — nbc=64 seçilirse resample neredeyse birebir olur).
    """
    dh = DOMAIN_EXTENT / nbc
    zoom_factors = [nbc / s for s in model_native.shape]
    model_fwd = scipy.ndimage.zoom(model_native, zoom_factors, order=1)   # order=1: bilineer

    half = DOMAIN_EXTENT / 2
    x_c = np.linspace(-half + dh / 2, half - dh / 2, nbc)
    y_c = np.linspace(-half + dh / 2, half - dh / 2, nbc)
    z_c = np.linspace(dh / 2, DOMAIN_EXTENT - dh / 2, nbc)

    obs_1d = np.linspace(-half, half, 21)
    obs_x, obs_y = np.meshgrid(obs_1d, obs_1d)

    grids = {"x_c": x_c, "y_c": y_c, "z_c": z_c, "obs_x": obs_x, "obs_y": obs_y, "half": half, "nbc": nbc, "dh": dh}
    return model_fwd, grids


# ── Yardımcılar: tekil fizik motoru çağrıları (run_physics + joint_inversion ORTAK) ──
def forward_grav(model_fwd: np.ndarray, grids: dict, engine_mode: str = "prism") -> np.ndarray:
    """(21,21) gz değerleri döndürür. engine_mode: 'prism' | 'fvm'"""
    density_contrast = model_fwd * DENSITY_SCALE
    if engine_mode == "fvm" and FVM_AVAILABLE:
        eng_g = PrismGravityForwardFVM(pad_cells=8)
        gz = eng_g.calculate(density_contrast, grids["x_c"], grids["y_c"], grids["z_c"],
                              grids["obs_x"], grids["obs_y"])
        return np.asarray(gz).reshape(21, 21)
    eng_g = gravity_prism.PrismGravityForward()
    gz = eng_g.calculate(density_contrast, grids["x_c"], grids["y_c"], grids["z_c"],
                         grids["obs_x"], grids["obs_y"])
    gz_np = gz.cpu().numpy() if hasattr(gz, "cpu") else np.array(gz)
    return np.asarray(gz_np).reshape(21, 21)


def forward_mag(model_fwd: np.ndarray, grids: dict, engine_mode: str = "prism") -> np.ndarray:
    """(21,21) toplam alan anomalisi döndürür. engine_mode: 'prism' | 'fvm'"""
    chi_contrast = model_fwd * SUSCEPT_SCALE
    if engine_mode == "fvm" and FVM_AVAILABLE:
        eng_m = PrismMagneticForwardFVM(inc_deg=60.0, dec_deg=5.0, b0_nt=47000.0, pad_cells=16)
        dt = eng_m.calculate(chi_contrast, grids["x_c"], grids["y_c"], grids["z_c"],
                              grids["obs_x"], grids["obs_y"])
        return np.asarray(dt).reshape(21, 21)
    eng_m = magnetic_prism.PrismMagneticForward(inc_deg=60.0, dec_deg=5.0, b0_nt=47000.0)
    dt = eng_m.calculate(chi_contrast, grids["x_c"], grids["y_c"], grids["z_c"],
                         grids["obs_x"], grids["obs_y"])
    dt_np = dt.cpu().numpy() if hasattr(dt, "cpu") else np.array(dt)
    return np.asarray(dt_np).reshape(21, 21)


def forward_csamt(model_fwd: np.ndarray, grids: dict):
    """Ham görünür özdirenç çıktısı + (21,21) istasyon-ortalamalı özet döndürür.

    NOT: csamt_1d motorunun tam çıktı şekli (freq × istasyon ya da istasyon × freq)
    projeye özgü olduğundan burada savunmacı (defensive) bir şekil çözümlemesi
    yapılıyor; motor değişse bile kırılmaması hedefleniyor.
    """
    nbc = grids.get("nbc", NBC_FORWARD)
    dh = grids.get("dh", DH_FORWARD)

    res_host, res_ratio = RES_HOST, RES_RATIO
    resistivity = res_host * np.power(res_ratio, model_fwd)

    res_columns = resistivity.reshape(nbc, nbc, nbc)
    obs_1d_idx = np.linspace(0, nbc - 1, 21, dtype=int)
    res_stn = res_columns[np.ix_(obs_1d_idx, obs_1d_idx, np.arange(nbc))]
    res_2d = res_stn.reshape(-1, nbc)   # (441, nbc)

    freqs = np.logspace(4, -1, 5)
    thicknesses = np.ones(nbc - 1) * dh

    eng_c = csamt_1d.CSAMT1DForward()
    result_csamt = eng_c.calculate(freqs, thicknesses, res_2d, track_gradients=False)

    # TUPLE-SAFE paketleme: motor (app_res, phase) tuple döndürebilir
    app_res = result_csamt[0] if isinstance(result_csamt, (list, tuple)) else result_csamt
    app_res_np = np.asarray(app_res.cpu().numpy() if hasattr(app_res, "cpu") else app_res)

    n_stations = 441
    if app_res_np.ndim == 1 and app_res_np.size == n_stations:
        station_mean = app_res_np.reshape(21, 21)
    elif app_res_np.ndim == 2 and n_stations in app_res_np.shape:
        station_axis = app_res_np.shape.index(n_stations)
        other_axis = 1 - station_axis
        station_mean = app_res_np.mean(axis=other_axis).reshape(21, 21)
    else:
        # Şekil çözümlenemedi: güvenli düşüş (scalar yayma)
        station_mean = np.full((21, 21), float(np.mean(app_res_np)))

    return app_res_np, station_mean


# ── Gradyan-koruyan (torch) forward sarmalayıcıları — SADECE _run_gradient_joint_inversion içinde kullanılır.
# Mevcut numpy forward_grav/forward_mag/forward_csamt (run_physics + eski SPSA yolu) DEĞİŞTİRİLMEDİ.
def forward_grav_t(model_fwd_t: torch.Tensor, grids: dict) -> torch.Tensor:
    density_contrast = model_fwd_t * DENSITY_SCALE
    eng_g = gravity_prism.PrismGravityForward()
    gz = eng_g.calculate(density_contrast, grids["x_c"], grids["y_c"], grids["z_c"],
                          grids["obs_x"], grids["obs_y"], track_gradients=True)
    return gz.reshape(21, 21)


def forward_mag_t(model_fwd_t: torch.Tensor, grids: dict) -> torch.Tensor:
    chi_contrast = model_fwd_t * SUSCEPT_SCALE
    eng_m = magnetic_prism.PrismMagneticForward(inc_deg=60.0, dec_deg=5.0, b0_nt=47000.0)
    dt = eng_m.calculate(chi_contrast, grids["x_c"], grids["y_c"], grids["z_c"],
                          grids["obs_x"], grids["obs_y"], track_gradients=True)
    return dt.reshape(21, 21)


def forward_csamt_t(model_fwd_t: torch.Tensor, grids: dict):
    nbc = grids.get("nbc", NBC_FORWARD)
    dh = grids.get("dh", DH_FORWARD)

    resistivity = RES_HOST * (RES_RATIO ** model_fwd_t)

    res_columns = resistivity.reshape(nbc, nbc, nbc)
    obs_1d_idx = np.linspace(0, nbc - 1, 21, dtype=int)
    # np.ix_ yerine ardışık indeksleme: hem numpy hem torch tensörlerinde aynı şekilde çalışır.
    res_stn = res_columns[obs_1d_idx][:, obs_1d_idx, :]
    res_2d = res_stn.reshape(-1, nbc)

    freqs = np.logspace(4, -1, 5)
    thicknesses = np.ones(nbc - 1) * dh

    eng_c = csamt_1d.CSAMT1DForward()
    app_res, phase = eng_c.calculate(freqs, thicknesses, res_2d, track_gradients=True)

    n_stations = res_2d.shape[0]
    if app_res.dim() == 1 and app_res.numel() == n_stations:
        station_mean = app_res.reshape(21, 21)
    elif app_res.dim() == 2 and n_stations in app_res.shape:
        station_axis = list(app_res.shape).index(n_stations)
        other_axis = 1 - station_axis
        station_mean = app_res.mean(dim=other_axis).reshape(21, 21)
    else:
        station_mean = app_res.mean().expand(21, 21)

    return app_res, station_mean


# ── Sağlık kontrolü ──────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "3.0"}


# ── Veri seti yönetimi (yükle / listele / sil) ───────────────────────────────
@app.get("/api/data/list")
def list_datasets():
    files = []
    for fn in sorted(os.listdir(UPLOAD_DIR)):
        if not fn.endswith(".npy"):
            continue
        full = os.path.join(UPLOAD_DIR, fn)
        size_kb = round(os.path.getsize(full) / 1024, 1)
        shape = None
        kind = "unknown"
        try:
            arr = np.load(full, mmap_mode="r")
            shape = list(arr.shape)
            kind = _classify_dataset_shape(shape)
        except Exception:
            shape = None
        files.append({"filename": fn, "size_kb": size_kb, "shape": shape, "kind": kind})
    return {"files": files, "default_available": os.path.exists(os.path.join(UPLOAD_DIR, DEFAULT_DATASET))}


@app.post("/api/data/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Sadece .npy dosyaları kabul edilir.")

    safe_name = _safe_npy_name(file.filename)
    dest = os.path.join(UPLOAD_DIR, safe_name)

    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    # Bütünlük kontrolü: gerçekten geçerli bir .npy mi?
    try:
        arr = np.load(dest, mmap_mode="r")
        shape = list(arr.shape)
        kind = _classify_dataset_shape(shape)
    except Exception as e:
        os.remove(dest)
        raise HTTPException(status_code=400, detail=f"Geçersiz .npy dosyası: {e}")

    return {"filename": safe_name, "shape": shape, "kind": kind, "size_kb": round(len(contents) / 1024, 1)}


@app.delete("/api/data/{filename}")
def delete_dataset(filename: str):
    safe_name = _safe_npy_name(filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")
    os.remove(path)
    return {"deleted": safe_name}


# ── Ana fizik hesaplama uç noktası ───────────────────────────────────────────
@app.post("/api/run-physics-engine")
def run_physics(req: SimulationRequest):
    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids = resample_to_forward(model_native)

    results = {}

    if req.grav_active:
        try:
            gz_np = forward_grav(model_fwd, grids, engine_mode=getattr(req,'engine_mode','prism'))
            results["Gravite"] = float(gz_np.max())
        except Exception as e:
            results["Gravite_hata"] = str(e)

    if req.mag_active:
        try:
            dt_np = forward_mag(model_fwd, grids, engine_mode=getattr(req,'engine_mode','prism'))
            results["Manyetik"] = float(dt_np.max())
        except Exception as e:
            results["Manyetik_hata"] = str(e)

    if req.csamt_active:
        try:
            app_res_np, _ = forward_csamt(model_fwd, grids)
            results["CSAMT"] = float(np.mean(app_res_np))
        except Exception as e:
            results["CSAMT_hata"] = str(e)

    return {
        "results": results,
        "model_data": model_fwd.tolist(),
        "selected_index": req.selected_index if used_path else -1,
        "dataset_used": os.path.basename(used_path) if used_path else "demo_sentetik",
        "meta": {
            "grid_size": NBC_FORWARD,
            "domain_m": DOMAIN_EXTENT,
            "dh_m": DH_FORWARD,
        },
    }


# ── Joint Inversion (SPSA tabanlı, petrofizikçe bağlı ortak model) ───────────
def _joint_objective(m: np.ndarray, grids: dict, weights: dict, active: dict,
                      d_obs: dict, reg_lambda: float):
    """Ağırlıklı veri uyumsuzluğu (misfit) + düzgünlük regularizasyonu."""
    parts = {}
    total = 0.0

    if active.get("grav"):
        gz = forward_grav(m, grids)
        val = weights.get("grav", 1.0) * float(np.mean((gz - d_obs["grav"]) ** 2))
        parts["grav"] = val
        total += val

    if active.get("mag"):
        dt = forward_mag(m, grids)
        val = weights.get("mag", 1.0) * float(np.mean((dt - d_obs["mag"]) ** 2))
        parts["mag"] = val
        total += val

    if active.get("csamt"):
        _, stn = forward_csamt(m, grids)
        val = weights.get("csamt", 1.0) * float(np.mean((stn - d_obs["csamt"]) ** 2))
        parts["csamt"] = val
        total += val

    lap = scipy.ndimage.laplace(m)
    reg = reg_lambda * float(np.mean(lap ** 2))
    total += reg
    return total, parts, reg


def _run_spsa_joint_inversion(shape, grids, weights, active, d_obs, n_iter: int,
                               reg_lambda: float, seed: int = 42):
    """Simultaneous Perturbation Stochastic Approximation (SPSA).

    Neden SPSA: grav/mag/csamt motorlarının analitik gradyanı (adjoint) bu projede
    mevcut değil; sonlu-farklar (finite-difference) gradyanı 4096 parametre için
    parametre başına 1 forward çağrısı gerektirir (çok pahalı). SPSA ise
    iterasyon başına yalnızca 2 forward-set çağrısıyla (m+ ve m-) TÜM parametreler
    için gradyan kestirimi üretir — bu tip kara-kutu (black-box) çok-yöntemli
    ters çözüm problemleri için standart ve pratik bir seçimdir.

    ÖNEMLİ (kalibrasyon): misfit ölçeği veri kaynağına göre devasa farklılık
    gösterir (self-forward sentetik d_obs ~1e-3, gerçek X_mag_grav/X_csamt
    verisiyle ~1e5). Sabit bir 'a' adım katsayısı bu iki durumda ya etkisiz
    kalır ya da parametreleri anında 0/1 sınırına fırlatıp yakınsamayı
    engelliyordu (gözlemlenen: misfit iterasyonlar boyunca yatay/salınımlı
    kalıyordu). Çözüm: Spall (1998)'in önerdiği gibi, birkaç PİLOT gradyan
    kestiriminin ortalama büyüklüğünü ölçüp 'a'yı, ilk iterasyonun parametre
    uzayında hedeflenen (target_step) kadar değişim yaratacağı şekilde geriye
    çözüyoruz — ölçekten tamamen bağımsız, kendi kendini kalibre eden SPSA.
    """
    rng = np.random.default_rng(seed)
    m = np.full(shape, 0.5)  # nötr başlangıç modeli

    c, A, alpha, gamma = 0.15, max(1.0, n_iter * 0.1), 0.602, 0.101

    # --- Pilot kalibrasyon: 'a' adım katsayısını misfit ölçeğine göre ayarla ---
    n_calib = 3
    target_step = 0.10  # ilk iterasyonda parametre uzayında hedeflenen değişim (0-1 aralığında)
    grad_mags = []
    for _ in range(n_calib):
        delta0 = rng.choice([-1.0, 1.0], size=m.size).reshape(shape)
        jp0, _, _ = _joint_objective(np.clip(m + c * delta0, 0.0, 1.0), grids, weights, active, d_obs, reg_lambda)
        jm0, _, _ = _joint_objective(np.clip(m - c * delta0, 0.0, 1.0), grids, weights, active, d_obs, reg_lambda)
        ghat0 = ((jp0 - jm0) / (2.0 * c)) * delta0
        grad_mags.append(float(np.mean(np.abs(ghat0))))
    mean_grad = max(float(np.mean(grad_mags)), 1e-12)
    a = target_step * ((1.0 + A) ** alpha) / mean_grad

    history = []

    for k in range(n_iter):
        ck = c / ((k + 1) ** gamma)
        ak = a / ((k + 1 + A) ** alpha)

        delta = rng.choice([-1.0, 1.0], size=m.size).reshape(shape)
        m_plus = np.clip(m + ck * delta, 0.0, 1.0)
        m_minus = np.clip(m - ck * delta, 0.0, 1.0)

        j_plus, _, _ = _joint_objective(m_plus, grids, weights, active, d_obs, reg_lambda)
        j_minus, _, _ = _joint_objective(m_minus, grids, weights, active, d_obs, reg_lambda)

        ghat = ((j_plus - j_minus) / (2.0 * ck)) * delta
        m = np.clip(m - ak * ghat, 0.0, 1.0)

        j_cur, parts_cur, reg_cur = _joint_objective(m, grids, weights, active, d_obs, reg_lambda)
        entry = {"iter": k + 1, "misfit": j_cur, "reg": reg_cur}
        for key, val in parts_cur.items():
            entry[f"misfit_{key}"] = val
        history.append(entry)

    return m, history


def _reg_term_t(m: torch.Tensor, reg_lambda: float) -> torch.Tensor:
    """Normalize roughness regularizasyonu — model ölçeğine göre bağımsız."""
    lap = _laplacian_3d_t(m)
    return reg_lambda * torch.mean(lap ** 2) / (torch.mean(m ** 2) + 1e-8)


def _auto_lr(shape, grids: dict, active: dict, d_obs_t: dict,
             weights: dict, reg_lambda: float, device,
             n_probe: int = 5, target_delta: float = 0.005) -> float:
    """Otomatik lr kalibrasyonu — gradyan büyüklüğüne göre."""
    grad_mags = []
    for _ in range(n_probe):
        probe = torch.full(shape, -2.0, dtype=torch.float64, device=device, requires_grad=True)
        m_c = torch.sigmoid(probe)
        total = torch.zeros((), dtype=torch.float64, device=device)
        if active.get("grav") and "grav" in d_obs_t:
            gz = forward_grav_t(m_c, grids)
            total = total + weights.get("grav", 1.0) * _normalized_mse_t(gz, d_obs_t["grav"])
        if active.get("mag") and "mag" in d_obs_t:
            dt = forward_mag_t(m_c, grids)
            total = total + weights.get("mag", 1.0) * _normalized_mse_t(dt, d_obs_t["mag"])
        total = total + _reg_term_t(m_c, reg_lambda)
        total.backward()
        grad_mags.append(probe.grad.abs().mean().item())
    mean_grad = max(float(np.mean(grad_mags)), 1e-12)
    return float(np.clip(target_delta / mean_grad, 1e-4, 0.5))


def _laplacian_3d_t(m: torch.Tensor) -> torch.Tensor:
    mp = torch.nn.functional.pad(m.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1))
    lap = (
        mp[:, :, 2:, 1:-1, 1:-1] + mp[:, :, :-2, 1:-1, 1:-1] +
        mp[:, :, 1:-1, 2:, 1:-1] + mp[:, :, 1:-1, :-2, 1:-1] +
        mp[:, :, 1:-1, 1:-1, 2:] + mp[:, :, 1:-1, 1:-1, :-2] -
        6 * mp[:, :, 1:-1, 1:-1, 1:-1]
    )
    return lap.squeeze(0).squeeze(0)


def _normalized_mse_t(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """obs.std² ile normalize — veri ölçeğinden bağımsız [0,~4] aralığı."""
    return torch.mean((pred - obs) ** 2) / (obs.std() ** 2 + 1e-8)


def _run_gradient_joint_inversion(shape, grids: dict, weights: dict, active: dict,
                                   d_obs: dict, n_iter: int, reg_lambda: float,
                                   seed: int = 42, lr: float = 0.0):
    """Adam + autograd tabanlı joint inversion.

    Düzeltmeler (reg_lambda tarama v4 bulgularından):
    1. _normalized_mse_t: obs.std() ile normalize
    2. Başlangıç modeli: sigmoid(-2)≈0.12 (daha gerçekçi prior)
    3. Otomatik lr: _auto_lr() ile gradyan ölçeğine göre
    4. _reg_term_t: model ölçeğine göre normalize roughness
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    d_obs_t = {k: torch.tensor(v, dtype=torch.float64, device=device) for k, v in d_obs.items()}
    m_param = torch.full(shape, -2.0, dtype=torch.float64, device=device, requires_grad=True)

    if lr == 0.0:
        lr = _auto_lr(shape, grids, active, d_obs_t, weights, reg_lambda, device)

    optimizer = torch.optim.Adam([m_param], lr=lr)
    history = []

    for k in range(n_iter):
        optimizer.zero_grad()
        m_c = torch.sigmoid(m_param)
        parts = {}
        total = torch.zeros((), dtype=torch.float64, device=device)

        if active.get("grav"):
            gz = forward_grav_t(m_c, grids)
            val = weights.get("grav", 1.0) * _normalized_mse_t(gz, d_obs_t["grav"])
            parts["grav"] = val
            total = total + val

        if active.get("mag"):
            dt = forward_mag_t(m_c, grids)
            val = weights.get("mag", 1.0) * _normalized_mse_t(dt, d_obs_t["mag"])
            parts["mag"] = val
            total = total + val

        if active.get("csamt"):
            _, stn = forward_csamt_t(m_c, grids)
            val = weights.get("csamt", 1.0) * _normalized_mse_t(stn, d_obs_t["csamt"])
            parts["csamt"] = val
            total = total + val

        reg = _reg_term_t(m_c, reg_lambda)
        total = total + reg

        total.backward()
        optimizer.step()

        entry = {"iter": k + 1, "misfit": total.item(), "reg": reg.item()}
        entry.update({f"misfit_{key}": val.item() for key, val in parts.items()})
        history.append(entry)

    m_final = torch.sigmoid(m_param).detach().cpu().numpy()
    return m_final, history


@app.post("/api/joint-inversion")
def joint_inversion(req: JointInversionRequest):
    active = {"grav": req.grav_active, "mag": req.mag_active, "csamt": req.csamt_active}
    if not any(active.values()):
        raise HTTPException(status_code=400, detail="En az bir yöntem aktif olmalı.")

    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_true, grids = resample_to_forward(model_native, nbc=req.nbc_forward)

    # "Gözlemlenen" veri (d_obs): mümkünse GERÇEK X_mag_grav/X_csamt dosyalarından
    # (data_factory.py çıktısı — aynı senaryo indeksiyle Y ile eşleşen gerçek forward
    # yanıtı), yoksa (X dosyası seçilmemişse) Y modelini kendimiz forward'layarak
    # sentetik/self-consistency d_obs üretiyoruz (demo/geliştirme modu).
    d_obs = {}
    used_real_grav_mag = False
    used_real_csamt = False

    gm_path = _resolve_dataset_path(req.dataset_grav_mag) if req.dataset_grav_mag else None
    if gm_path is not None:
        gm_sample = _load_indexed_npy(gm_path, req.selected_index)  # (21,21,2) -> [...,0]=mag,[...,1]=grav
        if gm_sample.ndim == 3 and gm_sample.shape[-1] == 2:
            if active["mag"]:
                d_obs["mag"] = gm_sample[..., 0]
            if active["grav"]:
                d_obs["grav"] = gm_sample[..., 1]
            used_real_grav_mag = True

    if not used_real_grav_mag:
        if active["grav"]:
            d_obs["grav"] = forward_grav(model_true, grids)
        if active["mag"]:
            d_obs["mag"] = forward_mag(model_true, grids)

    if active["csamt"]:
        csamt_path = _resolve_dataset_path(req.dataset_csamt) if req.dataset_csamt else None
        if csamt_path is not None:
            csamt_sample = _load_indexed_npy(csamt_path, req.selected_index)  # (21,21,n_freq)
            if csamt_sample.ndim == 3:
                d_obs["csamt"] = csamt_sample.mean(axis=-1)  # frekans ortalaması, forward_csamt ile TUTARLI
                used_real_csamt = True
        if not used_real_csamt:
            _, d_obs["csamt"] = forward_csamt(model_true, grids)

    # Korelasyon matrisi (Pearson) — aynı 21x21 istasyon ızgarasında hizalı alanlar
    correlation = {}
    keys = list(d_obs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_vec = d_obs[keys[i]].ravel()
            b_vec = d_obs[keys[j]].ravel()
            r = float(np.corrcoef(a_vec, b_vec)[0, 1])
            correlation[f"{keys[i]}_{keys[j]}"] = r

    m_inverted, history = _run_gradient_joint_inversion(
        shape=model_true.shape,
        grids=grids,
        weights=req.weights,
        active=active,
        d_obs=d_obs,
        n_iter=req.n_iter,
        reg_lambda=req.reg_lambda,
        seed=req.seed,
    )

    rmse_true_vs_inverted = float(np.sqrt(np.mean((model_true - m_inverted) ** 2)))

    return {
        "history": history,
        "correlation": correlation,
        "model_data": m_inverted.tolist(),
        "initial_misfit": history[0]["misfit"] if history else None,
        "final_misfit": history[-1]["misfit"] if history else None,
        "rmse_vs_true_model": rmse_true_vs_inverted,
        "dataset_used": os.path.basename(used_path) if used_path else "demo_sentetik",
        "d_obs_source": {
            "grav_mag": "gerçek (X_mag_grav)" if used_real_grav_mag else "self-forward (sentetik)",
            "csamt": "gerçek (X_csamt)" if used_real_csamt else "self-forward (sentetik)",
        },
        "meta": {"grid_size": req.nbc_forward, "domain_m": DOMAIN_EXTENT, "n_iter": req.n_iter},
    }




# ── Uncertainty Quantification ────────────────────────────────────────────────
class UncertaintyRequest(BaseModel):
    grav_active: bool = True
    mag_active: bool = True
    csamt_active: bool = False
    dataset: Optional[str] = None
    dataset_grav_mag: Optional[str] = None
    dataset_csamt: Optional[str] = None
    selected_index: int = 0
    nbc_forward: int = Field(default=16, ge=8, le=32)
    n_iter: int = Field(default=40, ge=10, le=120)
    reg_lambda: float = 0.001
    weights: Dict[str, float] = Field(default_factory=lambda: {"grav":1.0,"mag":1.0,"csamt":1.0})
    n_realizations: int = Field(default=5, ge=3, le=20)   # kaç farklı başlangıçtan çalıştır
    noise_level: float = Field(default=0.03, ge=0.0, le=0.2)  # d_obs'a eklenen gürültü oranı


@app.post("/api/uncertainty")
def uncertainty_quantification(req: UncertaintyRequest):
    """
    Jeolojik Belirsizlik Ölçümü — Intrepid GeoModeller MCMC yaklaşımının
    gradient tabanlı hafif analogu.

    Yöntem:
      n_realizations adet bağımsız inversion çalıştırılır:
        - Her biri farklı rastgele başlangıç noktası (seed)
        - Her biri d_obs'a küçük Gaussian gürültü eklenmiş veri

      Sonuçlar piksel bazında istatistiksel olarak birleştirilir:
        - mean_model   : ortalama model (en olası jeoloji)
        - std_model    : standart sapma (belirsizlik haritası)
        - p10_model    : P10 — kötümser senaryo
        - p90_model    : P90 — iyimser senaryo
        - cv_model     : Varyasyon katsayısı = std/mean (normalize belirsizlik)

    Yorum:
      - std yüksek bölge → verinin kısıtlayamadığı, belirsiz jeoloji
      - std düşük bölge  → tüm başlangıçlar aynı sonuca yakınsıyor, güvenilir
      - cv > 0.5        → o vokseldeki cevher varlığı güvenilir değil
    """
    active = {"grav": req.grav_active, "mag": req.mag_active, "csamt": req.csamt_active}
    if not any(active.values()):
        raise HTTPException(status_code=400, detail="En az bir yöntem aktif olmalı.")

    # Referans model ve gözlem verisi
    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_true, grids = resample_to_forward(model_native, nbc=req.nbc_forward)

    d_obs_base = {}
    gm_path = _resolve_dataset_path(req.dataset_grav_mag) if req.dataset_grav_mag else None
    if gm_path is not None:
        gm_sample = _load_indexed_npy(gm_path, req.selected_index)
        if gm_sample.ndim == 3 and gm_sample.shape[-1] == 2:
            if active["mag"]:  d_obs_base["mag"]  = gm_sample[..., 0]
            if active["grav"]: d_obs_base["grav"] = gm_sample[..., 1]
    if not d_obs_base:
        if active["grav"]: d_obs_base["grav"] = forward_grav(model_true, grids)
        if active["mag"]:  d_obs_base["mag"]  = forward_mag(model_true, grids)
    if active["csamt"]:
        csamt_path = _resolve_dataset_path(req.dataset_csamt) if req.dataset_csamt else None
        if csamt_path:
            cs = _load_indexed_npy(csamt_path, req.selected_index)
            if cs.ndim == 3: d_obs_base["csamt"] = cs.mean(axis=-1)
        if "csamt" not in d_obs_base:
            _, d_obs_base["csamt"] = forward_csamt(model_true, grids)

    # n_realizations çalıştır
    realizations = []
    histories    = []
    rng = np.random.default_rng(42)

    for i in range(req.n_realizations):
        seed_i = int(rng.integers(0, 99999))

        # Her realizasyona farklı gürültü ekle (data noise sensitivity)
        d_obs_i = {}
        for k, v in d_obs_base.items():
            noise = rng.normal(0, req.noise_level * float(np.std(v)), v.shape)
            d_obs_i[k] = v + noise

        m_i, hist_i = _run_gradient_joint_inversion(
            shape=model_true.shape,
            grids=grids,
            weights=req.weights,
            active=active,
            d_obs=d_obs_i,
            n_iter=req.n_iter,
            reg_lambda=req.reg_lambda,
            seed=seed_i,
        )
        realizations.append(m_i)
        histories.append({
            "seed": seed_i,
            "final_misfit": hist_i[-1]["misfit"],
            "convergence": [h["misfit"] for h in hist_i],
        })

    # İstatistiksel birleştirme
    stack = np.stack(realizations, axis=0)  # (n_real, nbc, nbc, nbc)

    mean_m = stack.mean(axis=0)
    std_m  = stack.std(axis=0)
    p10_m  = np.percentile(stack, 10, axis=0)
    p50_m  = np.percentile(stack, 50, axis=0)
    p90_m  = np.percentile(stack, 90, axis=0)
    cv_m   = std_m / (mean_m + 1e-6)   # varyasyon katsayısı

    # Özet istatistikler
    high_conf_voxels = int((cv_m < 0.3).sum())   # cv<0.3 → güvenilir
    low_conf_voxels  = int((cv_m > 0.7).sum())   # cv>0.7 → belirsiz
    total_voxels     = int(cv_m.size)

    # RMSE — her realizasyon için true modele karşı
    rmse_list = [float(np.sqrt(np.mean((model_true - m)**2))) for m in realizations]

    return {
        "mean_model"  : mean_m.tolist(),
        "std_model"   : std_m.tolist(),
        "p10_model"   : p10_m.tolist(),
        "p50_model"   : p50_m.tolist(),
        "p90_model"   : p90_m.tolist(),
        "cv_model"    : cv_m.tolist(),
        "realizations": [m.tolist() for m in realizations],
        "histories"   : histories,
        "summary": {
            "n_realizations"  : req.n_realizations,
            "noise_level"     : req.noise_level,
            "mean_final_misfit": float(np.mean([h["final_misfit"] for h in histories])),
            "std_final_misfit" : float(np.std([h["final_misfit"]  for h in histories])),
            "mean_rmse"       : float(np.mean(rmse_list)),
            "std_rmse"        : float(np.std(rmse_list)),
            "high_conf_pct"   : round(high_conf_voxels / total_voxels * 100, 1),
            "low_conf_pct"    : round(low_conf_voxels  / total_voxels * 100, 1),
            "rmse_per_real"   : rmse_list,
        },
        "dataset_used": os.path.basename(used_path) if used_path else "demo_sentetik",
        "meta": {"grid_size": req.nbc_forward, "n_realizations": req.n_realizations},
    }


# ── Harmonica Entegrasyonu ─────────────────────────────────────────────────────
try:
    from engines.harmonica_validation import validate_gravity, validate_magnetic, generate_synthetic_anomaly
    HARMONICA_AVAILABLE = True
except ImportError:
    HARMONICA_AVAILABLE = False


class HarmonicaValidateRequest(BaseModel):
    dataset: Optional[str] = None
    selected_index: int = 0
    check_gravity: bool = True
    check_magnetic: bool = True


class SyntheticRequest(BaseModel):
    nx: int = Field(default=16, ge=4, le=64)
    ny: int = Field(default=16, ge=4, le=64)
    nz: int = Field(default=16, ge=4, le=64)
    n_bodies: int = Field(default=2, ge=1, le=8)
    domain_m: float = Field(default=480.0, gt=0)
    seed: int = 42


@app.get("/api/harmonica/status")
def harmonica_status():
    return {"available": HARMONICA_AVAILABLE}


@app.post("/api/harmonica/validate")
def harmonica_validate(req: HarmonicaValidateRequest):
    if not HARMONICA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Harmonica kurulu değil. 'pip install harmonica verde' çalıştırın.")

    model_native, used = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids = resample_to_forward(model_native)

    report = {}

    if req.check_gravity:
        density_contrast = model_fwd * DENSITY_SCALE
        gz_np = forward_grav(model_fwd, grids)
        res = validate_gravity(
            density_contrast,
            grids["x_c"], grids["y_c"], grids["z_c"],
            grids["obs_x"], grids["obs_y"],
            our_gz=gz_np,
        )
        report["gravity"] = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in res.items()}

    if req.check_magnetic:
        chi = model_fwd * SUSCEPT_SCALE
        dt_np = forward_mag(model_fwd, grids)
        res = validate_magnetic(
            chi,
            grids["x_c"], grids["y_c"], grids["z_c"],
            grids["obs_x"], grids["obs_y"],
            our_dt=dt_np,
        )
        report["magnetic"] = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in res.items()}

    return {"report": report, "dataset_used": os.path.basename(used) if used else "demo_sentetik"}


@app.post("/api/harmonica/generate-synthetic")
def harmonica_generate_synthetic(req: SyntheticRequest):
    if not HARMONICA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Harmonica kurulu değil. 'pip install harmonica verde' çalıştırın.")

    result = generate_synthetic_anomaly(
        nx=req.nx, ny=req.ny, nz=req.nz,
        n_bodies=req.n_bodies,
        domain_m=req.domain_m,
        seed=req.seed,
    )

    return {
        "geometry": result["geometry"].tolist(),
        "gz_mgal": result["gz_mgal"].tolist(),
        "tmi_nt": result["tmi_nt"].tolist(),
        "body_params": result["body_params"],
        "grid": {"nx": req.nx, "ny": req.ny, "nz": req.nz, "domain_m": req.domain_m},
    }


# ── Referans Geometri Üretici ──────────────────────────────────────────────────
class SyntheticGeomRequest(BaseModel):
    geom_type: str = "beylikova_vein"
    nbc: int = Field(default=32, ge=8, le=64)
    dip_deg: float = Field(default=60.0, ge=0, le=90)
    depth_top_m: float = Field(default=30.0, ge=0)
    depth_bot_m: float = Field(default=400.0, ge=10)
    width_m: float = Field(default=60.0, ge=5)
    add_breccia: bool = True
    add_halo: bool = True
    seed: int = 42


class SaveGeometryRequest(BaseModel):
    model_data: List           # 3D liste (nbc × nbc × nbc)
    filename: str = "Y_generated.npy"


@app.post("/api/save-geometry")
def save_geometry(req: SaveGeometryRequest):
    """
    generate-geometry çıktısını uploads/ klasörüne .npy olarak kaydeder.
    Kaydedilen dosya artık run-physics-engine ve joint-inversion tarafından
    dataset parametresiyle kullanılabilir — Y_... prefix'i dataset listesinde görünür.
    """
    safe = _safe_npy_name(req.filename)
    if not safe.startswith("Y_"):
        safe = "Y_" + safe   # dataset listesinde Y_ altında görünmesi için

    arr = np.array(req.model_data, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[np.newaxis]   # (1, nbc, nbc, nbc) — tek örnek

    dest = os.path.join(UPLOAD_DIR, safe)
    np.save(dest, arr)

    return {
        "filename": safe,
        "shape": list(arr.shape),
        "size_kb": round(os.path.getsize(dest) / 1024, 1),
    }


@app.post("/api/generate-geometry")
def generate_geometry(req: SyntheticGeomRequest):
    """
    Bilinen prospeksiyon geometrisi oluştur — rastgele değil, kontrollü referans model.

    geom_type:
      beylikova_vein : KB-GD gidişli hidrotermal damar + breş (Beylikova REE-F-Ba-Th analogu)
      pipe           : Düşey borumsu yapı (kimberlite, Cu-Mo porfiri)
      lens           : Yatay elipsoidal mercek (SEDEX, VMS, tabaka uyumlu Au-Ag)
      stratabound    : Katmana bağlı yatay tabaka (Cu-Co, PGE, Zn-Pb)

    NOT: Y_beylikova.npy içindeki 2000 senaryo eğitim verisi olarak kullanılıyor.
    Bu endpoint eğitim verisinden BAĞIMSIZ, kullanıcı tarafından tam kontrol edilebilen
    tek bir referans geometri üretir — 'kesit indeksi' rastgele seçim yerine bunu kullan.
    """
    rng = np.random.default_rng(req.seed)
    nbc = req.nbc
    dh  = DOMAIN_EXTENT / nbc
    geom = np.zeros((nbc, nbc, nbc), dtype=np.float32)
    half = nbc // 2
    z_top = int(req.depth_top_m / dh)
    z_bot = min(nbc - 1, int(req.depth_bot_m / dh))
    w_vox = max(1.0, req.width_m / dh)
    dip   = np.radians(max(1.0, req.dip_deg))

    if req.geom_type == "beylikova_vein":
        for z in range(z_top, z_bot + 1):
            dip_shift = (z - z_top) / np.tan(dip)
            xc = half + int(dip_shift * 0.5)
            yc = half
            width = max(1.0, w_vox * (1 - (z - z_top) * 0.004))
            for x in range(nbc):
                for y in range(nbc):
                    dx, dy = (x - xc) / width, (y - yc) / (width * 1.5)
                    r = np.sqrt(dx**2 + dy**2)
                    if r < 1.0:
                        geom[x, y, z] = max(geom[x, y, z], float(1.0 - r ** 0.7))
        if req.add_breccia:
            breccia_depth = z_top + (z_bot - z_top) // 3
            for z in range(z_top, breccia_depth):
                for _ in range(10):
                    bx = int(rng.integers(half - 6, half + 6))
                    by = int(rng.integers(half - 5, half + 5))
                    r  = rng.uniform(1.0, 2.5)
                    s  = rng.uniform(0.4, 0.85)
                    for x in range(max(0, bx - 4), min(nbc, bx + 4)):
                        for y in range(max(0, by - 4), min(nbc, by + 4)):
                            d = np.sqrt((x - bx)**2 + (y - by)**2) / r
                            if d < 1.0:
                                geom[x, y, z] = max(geom[x, y, z], s * (1 - d ** 0.8))

    elif req.geom_type == "pipe":
        r_vox = w_vox / 2
        for x in range(nbc):
            for y in range(nbc):
                d = np.sqrt((x - half)**2 + (y - half)**2) / r_vox
                if d < 1.0:
                    val = float(1.0 - d ** 0.6)
                    geom[x, y, z_top:z_bot + 1] = np.maximum(
                        geom[x, y, z_top:z_bot + 1], val)

    elif req.geom_type == "lens":
        z_mid = (z_top + z_bot) // 2
        thick = max(1, (z_bot - z_top) // 2)
        for x in range(nbc):
            for y in range(nbc):
                dr = np.sqrt((x - half)**2 + (y - half)**2) / (nbc * 0.35)
                for z in range(z_top, z_bot + 1):
                    dz = abs(z - z_mid) / thick
                    d  = np.sqrt(dr**2 + dz**2)
                    if d < 1.0:
                        geom[x, y, z] = max(geom[x, y, z], float(1 - d ** 0.8))

    elif req.geom_type == "stratabound":
        z_mid = (z_top + z_bot) // 2
        thick = max(1, (z_bot - z_top) // 2)
        for x in range(nbc):
            for y in range(nbc):
                for z in range(z_top, z_bot + 1):
                    dz   = abs(z - z_mid) / thick
                    edge = min(x, y, nbc - 1 - x, nbc - 1 - y) / (nbc * 0.15)
                    val  = (1 - dz) * (1 - min(1.0, edge))
                    if val > 0.05:
                        geom[x, y, z] = max(geom[x, y, z], float(val))

    if req.add_halo:
        halo = scipy.ndimage.gaussian_filter(geom, sigma=1.0)
        geom = np.maximum(geom, halo * 0.3)

    geom = np.clip(geom, 0, 1).astype(np.float32)

    descriptions = {
        "beylikova_vein":
            "KB-GD gidişli, GD-eğimli hidrotermal damar + üst breş zonu + alterasyon halo. "
            "Beylikova (Eskişehir) REE-F-Ba-Th yatağı tip analogu. "
            "Karbonatlı kayaçlarda gelişmiş, yüksek yoğunluk ve düşük özdirenç kontrast.",
        "pipe":
            "Düşey eksenli borumsu/silindirik yapı. "
            "Kimberlite (elmas), Cu-Mo porfiri, alkalik Cu-Au sistemleri için tipik. "
            "Gravite ve IP anomalisi odaklanmış, manyetik halo zayıf.",
        "lens":
            "Yatay elipsoidal mercek geometrisi. "
            "SEDEX (Zn-Pb-Ag), VMS (Cu-Zn), tabaka uyumlu Au-Ag yatakları için. "
            "Geniş gravite anomalisi, CSAMT/MT doğrudan derinlik kontrolü sağlar.",
        "stratabound":
            "Katmana bağlı yatay tabaka, kenar bölgelerde yoğunlaşma. "
            "Sedimantta Cu-Co (Kongo tipi), PGE reef, karbonatlarda Zn-Pb (Mississippi Valley) için. "
            "Düşük eğim açısı nedeniyle gravite ve manyetik ayrımı zordur.",
    }

    petrophys = {
        "density_contrast_gcm3": {
            "host":     2.70,
            "ore_max":  round(2.70 + 2.0 * float(geom.max()), 3),
            "formula":  "ρ(x) = 2.70 + 2.00 × f(x)   [g/cm³]",
            "note":     "Beylikova tipi: barit+REE-florit ağır mineral asemblajı"
        },
        "susceptibility_SI": {
            "host":     1e-4,
            "ore_max":  round(1e-4 + 3e-4 * float(geom.max()), 7),
            "formula":  "χ(x) = 1×10⁻⁴ + 3×10⁻⁴ × f(x)   [SI]",
            "note":     "Magnetit içeren alterasyon zonunda yüksek"
        },
        "resistivity_ohmm": {
            "host":     500.0,
            "ore_min":  round(500.0 * (0.10 ** float(geom.max())), 2),
            "formula":  "ρₑ(x) = 500 × 0.10^f(x)   [Ω·m]",
            "note":     "Grafitik/sülfürlü alterasyon → iletken"
        },
    }

    return {
        "model_data":  geom.tolist(),
        "shape":       list(geom.shape),
        "dh_m":        dh,
        "domain_m":    DOMAIN_EXTENT,
        "geom_type":   req.geom_type,
        "description": descriptions.get(req.geom_type, ""),
        "petrophys":   petrophys,
        "params": {
            "dip_deg":      req.dip_deg,
            "depth_top_m":  req.depth_top_m,
            "depth_bot_m":  req.depth_bot_m,
            "width_m":      req.width_m,
            "grid":         f"{nbc}³",
            "voxel_m":      dh,
        },
        "stats": {
            "max_value":     float(geom.max()),
            "ore_fraction":  float((geom > 0.5).mean()),
            "halo_fraction": float(((geom > 0.1) & (geom <= 0.5)).mean()),
            "active_voxels": int((geom > 0.05).sum()),
        }
    }


# ── Analiz Kayıt / Listeleme / Silme ─────────────────────────────────────────
import json as _json
import uuid as _uuid
import datetime as _dt

ANALYSES_DIR = "analyses"
os.makedirs(ANALYSES_DIR, exist_ok=True)


class AnalysisSaveRequest(BaseModel):
    name: str
    type: str
    dataset_used: Optional[str] = None
    settings: Optional[Dict] = None
    results: Optional[Dict] = None
    metrics: Optional[Dict] = None
    model_data: Optional[List] = None   # büyük — .npy olarak ayrı kaydedilir
    history: Optional[List] = None
    correlation: Optional[Dict] = None
    summary: Optional[Dict] = None


@app.get("/api/analyses")
def list_analyses():
    records = []
    for fn in sorted(os.listdir(ANALYSES_DIR), reverse=True):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(ANALYSES_DIR, fn), encoding="utf-8") as f:
                rec = _json.load(f)
            # model_data JSON'da tutulmuyor, sadece path
            records.append({k: v for k, v in rec.items() if k != "model_data"})
        except Exception:
            pass
    return {"analyses": records}


@app.post("/api/analyses")
def save_analysis(req: AnalysisSaveRequest):
    aid = str(_uuid.uuid4())[:8]
    now = _dt.datetime.now().isoformat()

    # model_data'yı ayrı .npy olarak kaydet (JSON şişmesini önler)
    model_npy_path = None
    if req.model_data:
        model_npy_path = os.path.join(ANALYSES_DIR, f"{aid}_model.npy")
        np.save(model_npy_path, np.array(req.model_data, dtype=np.float32))

    record = {
        "id": aid,
        "name": req.name,
        "type": req.type,
        "created_at": now,
        "dataset_used": req.dataset_used,
        "settings": req.settings,
        "results": req.results,
        "metrics": req.metrics,
        "model_npy_path": model_npy_path,  # JSON'da path, veri ayrı
        "history": req.history,
        "correlation": req.correlation,
        "summary": req.summary,
    }
    path = os.path.join(ANALYSES_DIR, f"{aid}.json")
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(record, f, ensure_ascii=False)
    return {"id": aid, "created_at": now}


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str):
    if not re.match(r"^[A-Za-z0-9\-]+$", analysis_id):
        raise HTTPException(status_code=400, detail="Geçersiz analiz ID.")
    path = os.path.join(ANALYSES_DIR, f"{analysis_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Analiz bulunamadı.")
    with open(path, encoding="utf-8") as f:
        rec = _json.load(f)
    # model_data .npy'den oku
    npy = rec.get("model_npy_path")
    if npy and os.path.exists(npy):
        rec["model_data"] = np.load(npy).tolist()
    return rec


@app.delete("/api/analyses/{analysis_id}")
def delete_analysis(analysis_id: str):
    if not re.match(r"^[A-Za-z0-9\-]+$", analysis_id):
        raise HTTPException(status_code=400, detail="Geçersiz analiz ID.")
    path = os.path.join(ANALYSES_DIR, f"{analysis_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Analiz bulunamadı.")
    # model .npy'yi de sil
    try:
        with open(path, encoding="utf-8") as f:
            rec = _json.load(f)
        npy = rec.get("model_npy_path")
        if npy and os.path.exists(npy):
            os.remove(npy)
    except Exception:
        pass
    os.remove(path)
    return {"deleted": analysis_id}


# ── Veri Formatı Seçici ────────────────────────────────────────────────────────
@app.get("/api/data/formats")
def list_data_formats():
    """Desteklenen veri formatlarını ve beklenen sütunları döndürür."""
    return {"formats": SUPPORTED_FORMATS if RADIOMETRY_AVAILABLE else {
        "model_npy": {"extensions":[".npy"],"description":"GeoPINN 3D model grid","columns":[],"optional":[]}
    }}


@app.post("/api/data/detect-format")
async def detect_data_format(file: UploadFile = File(...)):
    """Yüklenen dosyanın formatını otomatik tespit eder."""
    content_bytes = await file.read(2048)
    try:
        header = content_bytes.decode('utf-8-sig', errors='replace')
    except Exception:
        header = ""
    fmt = detect_format(file.filename, header) if RADIOMETRY_AVAILABLE else "unknown"
    fmt_info = SUPPORTED_FORMATS.get(fmt, {}) if RADIOMETRY_AVAILABLE else {}
    return {
        "filename":    file.filename,
        "detected":    fmt,
        "description": fmt_info.get("description", "Bilinmeyen format"),
        "expected_columns": fmt_info.get("columns", []),
        "optional_columns": fmt_info.get("optional", []),
        "confidence":  "high" if fmt != "unknown" else "low",
    }


# ── Radyometri & Isı Akışı ────────────────────────────────────────────────────
class RadiometryRequest(BaseModel):
    dataset: Optional[str] = None          # Y_... model küpü (U/Th/K normalize [0,1])
    selected_index: int = 0
    # Petrofizik ölçekleme (modeli gerçek konsantrasyona dönüştür)
    u_background_ppm:  float = 3.0
    u_ore_ppm:         float = 15.0
    th_background_ppm: float = 12.0
    th_ore_ppm:        float = 60.0
    k_background_pct:  float = 2.5
    k_ore_pct:         float = 4.5
    # Isı akışı parametreleri
    k_thermal:  float = 2.5    # W/(m·K) — kireçtaşı host
    T_surface:  float = 15.0   # °C
    T_base:     float = 65.0   # °C
    # Hangi çıktılar isteniyor
    compute_heat_flow:   bool = True
    compute_radiometry:  bool = True
    compute_ree_index:   bool = True


@app.post("/api/radiometry/forward")
def radiometry_forward(req: RadiometryRequest):
    """
    Cevher modeli → U/Th/K konsantrasyonu → radyometri + ısı akışı.

    Model [0,1] normalize değerleri arka plan + cevher arasında lineer interpole eder.
    """
    if not RADIOMETRY_AVAILABLE:
        raise HTTPException(status_code=503,
            detail="Radyometri modülleri yüklenemedi. engines/radiometry.py ve heat_flow_fvm.py kontrol edin.")

    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids = resample_to_forward(model_native)

    n = model_fwd.shape[0]
    f = model_fwd  # [0,1]

    # Model → konsantrasyon
    u_grid  = req.u_background_ppm  + f * (req.u_ore_ppm  - req.u_background_ppm)
    th_grid = req.th_background_ppm + f * (req.th_ore_ppm - req.th_background_ppm)
    k_grid  = req.k_background_pct  + f * (req.k_ore_pct  - req.k_background_pct)

    result = {
        "dataset_used": os.path.basename(used_path) if used_path else "demo_sentetik",
        "grid_size": n,
        "petrophys": {
            "u_range_ppm":  [req.u_background_ppm, req.u_ore_ppm],
            "th_range_ppm": [req.th_background_ppm, req.th_ore_ppm],
            "k_range_pct":  [req.k_background_pct, req.k_ore_pct],
        }
    }

    if req.compute_radiometry:
        rad = forward_radiometry_surface(u_grid, th_grid, k_grid)
        result["radiometry"] = {
            "TC_cps":     rad["TC_cps"].tolist(),
            "U_cps":      rad["U_cps"].tolist(),
            "Th_cps":     rad["Th_cps"].tolist(),
            "K_cps":      rad["K_cps"].tolist(),
            "eU_ppm":     rad["eU_ppm"].tolist(),
            "Th_U":       np.nan_to_num(rad["Th_U"], nan=0).tolist(),
            "dose_nGy_h": rad["dose_nGy_h"].tolist(),
            "stats": {
                "TC_max":   float(rad["TC_cps"].max()),
                "TC_mean":  float(rad["TC_cps"].mean()),
                "Th_U_max": float(np.nanmax(rad["Th_U"])),
                "dose_max": float(rad["dose_nGy_h"].max()),
            }
        }

    if req.compute_ree_index:
        idx = ree_alteration_index(u_grid, th_grid, k_grid)
        result["ree_index"] = {
            "composite_score":  idx["composite_score"].tolist(),
            "ree_target_prob":  idx["ree_target_prob"].tolist(),
            "Th_U_ratio":       np.nan_to_num(idx["Th_U_ratio"], nan=0).tolist(),
            "eU_ppm":           idx["eU_ppm"].tolist(),
            "stats": {
                "max_prob":        float(idx["ree_target_prob"].max()),
                "high_prob_cells": int((idx["ree_target_prob"] > 0.5).sum()),
                "Th_U_mean":       float(np.nanmean(idx["Th_U_ratio"])),
                "interpretation":  "Yüksek REE hedef olasılığı" if idx["ree_target_prob"].max() > 0.6
                                   else "Orta/düşük REE sinyali",
            }
        }

    if req.compute_heat_flow:
        eng = HeatFlowFVM(
            k_thermal=req.k_thermal,
            T_surface=req.T_surface,
            T_base=req.T_base,
        )
        hf = eng.calculate(
            u_grid, th_grid, k_grid,
            grids["x_c"], grids["y_c"], grids["z_c"],
            depth_indices=[0, n//4, n//2],
        )
        result["heat_flow"] = {
            "surface_flux_mw_m2": hf["heat_flux_mw_m2"],
            "T_field_degC":       hf["T_field"],
            "Q_field_W_m3":       hf["Q_field"],
            "stats":              hf["stats"],
        }

    return result


@app.post("/api/radiometry/upload-field-data")
async def upload_radiometry_field_data(
    file: UploadFile = File(...),
    format_hint: str = "auto",
):
    """
    Saha radyometri/SP/IP/sismik verisini yükler, parse eder ve özet döndürür.
    format_hint: 'auto' | 'radiometry_csv' | 'sp_csv' | 'ip_csv' | ...
    """
    if not file.filename.endswith(('.csv','.xyz','.dat','.txt','.npy')):
        raise HTTPException(status_code=400,
            detail="Desteklenen formatlar: .csv, .xyz, .dat, .txt, .npy")

    import tempfile
    contents = await file.read()
    safe = _safe_npy_name(file.filename) if file.filename.endswith('.npy') else            re.sub(r'[^A-Za-z0-9_\-\.]', '_', file.filename)
    dest = os.path.join(UPLOAD_DIR, safe)
    with open(dest, 'wb') as f:
        f.write(contents)

    if not RADIOMETRY_AVAILABLE:
        return {"filename": safe, "size_kb": round(len(contents)/1024,1),
                "detected_format": "unknown", "note": "Radiometry modülü yüklü değil"}

    try:
        header = contents[:1024].decode('utf-8-sig', errors='replace')
        detected = detect_format(file.filename, header) if format_hint == 'auto' else format_hint
        data, fmt = read_field_data(dest, detected)
        stats = radiometry_stats(data) if fmt == "radiometry_csv" else {}
        return {
            "filename":        safe,
            "size_kb":         round(len(contents)/1024, 1),
            "detected_format": fmt,
            "n_points":        data.get("_n_points", 0),
            "columns":         [k for k in data if not k.startswith('_')],
            "stats":           stats,
            "format_info":     SUPPORTED_FORMATS.get(fmt, {}),
        }
    except Exception as e:
        return {"filename": safe, "error": str(e), "detected_format": "unknown"}


@app.get("/api/radiometry/status")
def radiometry_status():
    return {
        "available": RADIOMETRY_AVAILABLE,
        "supported_formats": list(SUPPORTED_FORMATS.keys()) if RADIOMETRY_AVAILABLE else [],
        "methods": {
            "forward_radiometry": "U/Th/K → Gammaray sayımı, Th/U oranı, eU",
            "heat_flow":          "U/Th/K → Radyojenik ısı üretimi → Yüzey ısı akışı (FVM)",
            "ree_index":          "Bileşik REE anomali skoru (Th/U, eU, K/Th)",
            "field_data_upload":  "Saha verisi yükleme (CSV/XYZ/DAT)",
        }
    }


# ── Seismic + Methods (eski frontend uyumluluğu) ─────────────────────────────
VP_CLASSES_TABLE = [
    {"id":1,"label":"Q1 — Toprak/Dolgu",         "vp_min":0,    "vp_max":800,  "color":"#8B5E3C","rqd":0, "ucs_mpa":0,  "drill_factor":0.3},
    {"id":2,"label":"Q2 — Ayrışmış Kaya",          "vp_min":800,  "vp_max":1800, "color":"#C8874A","rqd":10,"ucs_mpa":10, "drill_factor":0.5},
    {"id":3,"label":"Q3 — Kırıklı Kaya (Zayıf)",  "vp_min":1800, "vp_max":3000, "color":"#D4B84A","rqd":30,"ucs_mpa":30, "drill_factor":0.7},
    {"id":4,"label":"Q4 — Kırıklı Sağlam",         "vp_min":3000, "vp_max":4000, "color":"#5A8A3C","rqd":60,"ucs_mpa":60, "drill_factor":1.0},
    {"id":5,"label":"Q5 — Sağlam Granit/Kireç",   "vp_min":4000, "vp_max":5500, "color":"#2E6B8A","rqd":85,"ucs_mpa":120,"drill_factor":1.8},
    {"id":6,"label":"Q6 — Sert Bazalt/Kuvarsit",   "vp_min":5500, "vp_max":9999, "color":"#1C2E5E","rqd":95,"ucs_mpa":200,"drill_factor":3.0},
]

@app.get("/api/seismic/vp-classes")
def seismic_vp_classes():
    return {"classes": VP_CLASSES_TABLE, "engine_available": True}


@app.get("/api/methods")
def list_methods():
    return {
        "methods": [
            {"id":"gravity",    "label":"Gravite",            "available":True,  "engine":"gravity_prism (Nagy analitik)",         "gpu":True},
            {"id":"magnetic",   "label":"Manyetik (TMI)",     "available":True,  "engine":"magnetic_prism (Bhattacharyya)",        "gpu":True},
            {"id":"csamt",      "label":"CSAMT / MT",         "available":True,  "engine":"csamt_1d (Wait özyineleme)",            "gpu":True},
            {"id":"joint_inv",  "label":"Joint Inversion",    "available":True,  "engine":"Adam + autograd",                       "gpu":True},
            {"id":"simpeg",     "label":"SimPEG Tikhonov",    "available":SIMPEG_AVAILABLE, "engine":"SimPEG L2 Gauss-Newton",    "gpu":False},
            {"id":"uncertainty","label":"Belirsizlik (UQ)",   "available":True,  "engine":"Monte Carlo realizasyonları",           "gpu":True},
            {"id":"fvm",        "label":"FVM (Poisson)",      "available":FVM_AVAILABLE,    "engine":"fvm_core scipy CG",         "gpu":False},
            {"id":"radiometry", "label":"Radyometri / Isı",   "available":RADIOMETRY_AVAILABLE, "engine":"heat_flow_fvm + radiometry","gpu":False},
            {"id":"harmonica",  "label":"Harmonica Doğrulama","available":HARMONICA_AVAILABLE,  "engine":"Fatiando a Terra",      "gpu":False},
        ]
    }


# ── FVM Motor Durumu & Karşılaştırma ──────────────────────────────────────────
# ── Data Fusion ────────────────────────────────────────────────────────────────
class DataFusionRequest(BaseModel):
    dataset:          Optional[str] = None
    selected_index:   int   = 0
    method:           str   = "gamma"   # weighted|gamma|and|or|index
    gamma:            float = 0.85
    # Hangi yöntemler dahil edilsin
    use_gravity:      bool  = True
    use_magnetic:     bool  = True
    use_csamt:        bool  = True
    use_ip:           bool  = False
    use_sp:           bool  = False
    use_radiometry:   bool  = False
    use_heat_flow:    bool  = False
    # Ağırlıklar (method=weighted için)
    w_gravity:        float = 0.15
    w_magnetic:       float = 0.15
    w_csamt:          float = 0.15
    w_ip:             float = 0.20
    w_sp:             float = 0.15
    w_radiometry:     float = 0.15
    w_heat_flow:      float = 0.05


@app.post("/api/fusion/composite")
def fusion_composite(req: DataFusionRequest):
    """Çok-yöntemli bileşik anomali skoru (Porwal et al. 2003)."""
    if not FUSION_AVAILABLE:
        raise HTTPException(status_code=503,
            detail="Fusion modülü yüklenemedi.")

    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids = resample_to_forward(model_native)
    import numpy as np

    # Forward hesapla — aktif yöntemler için
    grav_data = mag_data = csamt_data = ip_data = sp_data = None
    th_u_data = eu_data = hf_data = None

    if req.use_gravity and GRAV_AVAILABLE:
        try:
            eng = PrismGravityForward()
            r = eng.calculate(model_fwd, grids["x_c"], grids["y_c"], grids["z_c"])
            grav_data = np.array(r["gz_mgal"])
        except Exception as e:
            print(f"[fusion] gravity hatası: {e}")

    if req.use_magnetic and MAG_AVAILABLE:
        try:
            eng = PrismMagneticForward()
            r = eng.calculate(model_fwd, grids["x_c"], grids["y_c"], grids["z_c"])
            mag_data = np.array(r["tmi_nt"])
        except Exception as e:
            print(f"[fusion] magnetic hatası: {e}")

    if req.use_ip and IP_AVAILABLE:
        from engines.ip_forward import IPForwardMotor
        eng = IPForwardMotor()
        obs_x = np.linspace(0, DOMAIN_EXTENT, 21)
        r = eng.calculate(model_fwd, grids["x_c"], grids["z_c"], obs_x)
        ip_data = np.array(r["chargeability"])

    if req.use_sp and SP_AVAILABLE:
        eng = SPForwardMotor()
        r = eng.calculate(model_fwd, grids["x_c"], grids["y_c"], grids["z_c"])
        sp_data = np.array(r["sp_mv"])

    if req.use_radiometry and RADIOMETRY_AVAILABLE:
        u  = 3.0  + model_fwd * 12.0
        th = 12.0 + model_fwd * 48.0
        k  = 2.5  + model_fwd * 2.0
        rad = forward_radiometry_surface(u, th, k)
        th_u_data = np.nan_to_num(rad["Th_U"])
        eu_data   = rad["eU_ppm"]
        if req.use_heat_flow:
            from engines.heat_flow_fvm import HeatFlowFVM
            heng = HeatFlowFVM()
            hr = heng.calculate(u, th, k, grids["x_c"], grids["y_c"], grids["z_c"])
            hf_data = np.array(hr["heat_flux_mw_m2"])

    weights = {
        "gravity":   req.w_gravity,
        "magnetic":  req.w_magnetic,
        "csamt":     req.w_csamt,
        "ip":        req.w_ip,
        "sp":        req.w_sp,
        "radiometry":req.w_radiometry,
        "heat_flow": req.w_heat_flow,
    }

    eng_f = DataFusionEngine()
    result = eng_f.fuse(
        gravity_mgal      = grav_data,
        magnetic_nt       = mag_data,
        ip_chargeability  = ip_data,
        sp_mv             = sp_data,
        radiometry_th_u   = th_u_data,
        radiometry_eu_ppm = eu_data,
        heat_flow_mw_m2   = hf_data,
        method  = req.method,
        gamma   = req.gamma,
        weights = weights if req.method == "weighted" else None,
    )

    return {
        "dataset_used":    os.path.basename(used_path) if used_path else "demo",
        "composite_score": result["composite_score"],
        "method_scores":   result["method_scores"],
        "stats":           result["stats"],
    }


@app.get("/api/fusion/status")
def fusion_status():
    return {
        "available": FUSION_AVAILABLE,
        "methods":   ["weighted", "gamma", "and", "or", "index"],
        "reference": "Porwal et al. (2003), Bonham-Carter (1994)",
        "default_weights": DataFusionEngine.DEFAULT_WEIGHTS,
    }


# ── SP (Self-Potential) ───────────────────────────────────────────────────────
class SPForwardRequest(BaseModel):
    dataset:          Optional[str] = None
    selected_index:   int   = 0
    sigma_host:       float = 2e-3
    sigma_ore:        float = 0.1
    porosity_host:    float = 0.05
    porosity_ore:     float = 0.15
    w_electrokinetic: float = 1.0
    w_thermoelectric: float = 0.3
    w_electrochemical:float = 0.5
    use_heat_flow:    bool  = False


@app.post("/api/sp/forward")
def sp_forward(req: SPForwardRequest):
    """
    Öz-Potansiyel (SP) ileri modelleme.
    Elektrokinetik + Termoelektrik + Elektrokimyasal kuplaj.
    Referans: Revil & Leroy (2004), Sill (1983), Mendonça (2008).
    """
    if not SP_AVAILABLE:
        raise HTTPException(status_code=503,
            detail="SP modülü yüklenemedi. engines/sp_forward.py kontrol edin.")

    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids = resample_to_forward(model_native)

    motor = SPForwardMotor()
    result = motor.calculate(
        model_fwd,
        grids["x_c"], grids["y_c"], grids["z_c"],
        sigma_host=req.sigma_host,
        sigma_ore=req.sigma_ore,
        porosity_host=req.porosity_host,
        porosity_ore=req.porosity_ore,
        w_electrokinetic=req.w_electrokinetic,
        w_thermoelectric=req.w_thermoelectric,
        w_electrochemical=req.w_electrochemical,
    )

    return {
        "dataset_used": os.path.basename(used_path) if used_path else "demo_sentetik",
        "sp_mv":        result["sp_mv"],
        "sp_ek_mv":     result["sp_ek_mv"],
        "sp_te_mv":     result["sp_te_mv"],
        "sp_ec_mv":     result["sp_ec_mv"],
        "obs_x":        result["obs_x"],
        "obs_y":        result["obs_y"],
        "stats":        result["stats"],
        "parameters": {
            "sigma_host_Sm":  req.sigma_host,
            "sigma_ore_Sm":   req.sigma_ore,
            "weights": {
                "electrokinetic":  req.w_electrokinetic,
                "thermoelectric":  req.w_thermoelectric,
                "electrochemical": req.w_electrochemical,
            }
        }
    }


@app.get("/api/sp/status")
def sp_status():
    return {
        "available": SP_AVAILABLE,
        "mechanisms": {
            "electrokinetic":  "Streaming potential — hidrotermal akışkan (Revil & Leroy 2004)",
            "thermoelectric":  "Seebeck etkisi — sıcaklık gradyanı (Revil et al. 2012)",
            "electrochemical": "Battery model — sülfür mineralizasyonu (Sato & Mooney 1960)",
        },
        "reference": "Mendonça (2008), Geophysics 73(1), F33-F43",
        "beylikova_expected_mv": "-50 ile -300 mV (hidrotermal + sülfür)",
    }


# ── IP (Induced Polarization) ─────────────────────────────────────────────────
class IPForwardRequest(BaseModel):
    dataset:       Optional[str] = None
    selected_index: int = 0
    frequencies:   List[float] = [0.125, 0.5, 2.0, 8.0, 32.0]
    a_spacing:     float = 20.0
    n_max:         int   = 6
    # Petrofizik — host kaya
    rho0_host:     float = 500.0
    m_host:        float = 0.02
    tau_host:      float = 0.10
    c_host:        float = 0.50
    # Petrofizik — cevher/sülfür
    rho0_ore:      float = 15.0
    m_ore:         float = 0.55
    tau_ore:       float = 0.80
    c_ore:         float = 0.35


@app.post("/api/ip/forward")
def ip_forward(req: IPForwardRequest):
    """
    Cole-Cole IP ileri modelleme — dipole-dipole pseudosection.
    Referans: Pelton et al. (1978), Geophysics 43(3).
    """
    if not IP_AVAILABLE:
        raise HTTPException(status_code=503,
            detail="IP modülü yüklenemedi. engines/ip_forward.py kontrol edin.")

    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids = resample_to_forward(model_native)

    motor = IPForwardMotor()
    import numpy as np
    freqs = np.array(req.frequencies)
    obs_x = np.linspace(0, DOMAIN_EXTENT, 21)

    result = motor.calculate(
        model_fwd,
        grids["x_c"], grids["z_c"], obs_x,
        frequencies=freqs,
        a_spacing=req.a_spacing,
        n_max=req.n_max,
        rho0_host=req.rho0_host, m_host=req.m_host,
        tau_host=req.tau_host,  c_host=req.c_host,
        rho0_ore=req.rho0_ore,  m_ore=req.m_ore,
        tau_ore=req.tau_ore,    c_ore=req.c_ore,
    )

    return {
        "dataset_used":   os.path.basename(used_path) if used_path else "demo_sentetik",
        "rho_a_dc":       result["rho_a_dc"].tolist(),
        "chargeability":  result["chargeability"].tolist(),
        "phase_mrad":     result["phase_mrad"][:, :, 0].tolist(),  # İlk frekans
        "pseudo_x":       result["pseudo_x"].tolist(),
        "pseudo_z":       result["pseudo_z"].tolist(),
        "K_factors":      result["K_factors"].tolist(),
        "n_values":       result["n_values"].tolist(),
        "stats":          result["stats"],
        "parameters": {
            "a_spacing_m":   req.a_spacing,
            "n_max":         req.n_max,
            "frequencies_hz": req.frequencies,
            "cole_cole_host": {"rho0": req.rho0_host, "m": req.m_host,
                               "tau": req.tau_host,  "c": req.c_host},
            "cole_cole_ore":  {"rho0": req.rho0_ore,  "m": req.m_ore,
                               "tau": req.tau_ore,   "c": req.c_ore},
        }
    }


@app.get("/api/ip/status")
def ip_status():
    return {
        "available": IP_AVAILABLE,
        "method":    "Cole-Cole kompleks özdirenç, dipole-dipole array",
        "reference": "Pelton et al. (1978), Geophysics 43(3), 588-609",
        "parameters": ["rho0", "m (chargeability)", "tau (relaxation)", "c (frequency exponent)"],
        "beylikova_targets": {
            "host": "ρ₀=500 Ω·m, m=0.02 (kireçtaşı/şist)",
            "altered": "ρ₀=50 Ω·m, m=0.25 (sülfürlü alterasyon)",
            "ore": "ρ₀=15 Ω·m, m=0.55 (pirit/arsenopirit zonu)",
        }
    }


# ── Sismik Refraksiyon ───────────────────────────────────────────────────────
class SeismicRequest(BaseModel):
    dataset:          Optional[str] = None
    selected_index:   int   = 0
    shot_offset_max:  float = 480.0
    n_receivers:      int   = 24
    noise_pct:        float = 2.0
    vp_host:          float = 2500.0
    vp_ore:           float = 4500.0


@app.post("/api/seismic/forward")
def seismic_forward(req: SeismicRequest):
    """Sismik refraksiyon ileri modeli — t-x eğrisi ve Vp profili."""
    if not SEISMIC_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sismik modül yüklenemedi.")
    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids = resample_to_forward(model_native)
    motor = SeismicRefracMotor()
    result = motor.calculate(
        model_fwd, grids["x_c"], grids["z_c"],
        shot_offset_max=req.shot_offset_max,
        n_receivers=req.n_receivers,
        noise_pct=req.noise_pct,
        vp_host=req.vp_host,
        vp_ore=req.vp_ore,
    )
    return {"dataset_used": os.path.basename(used_path) if used_path else "demo", **result}


@app.get("/api/seismic/vp-classes")
def seismic_vp_classes():
    return {"classes": VP_CLASSES if SEISMIC_AVAILABLE else [], "engine_available": SEISMIC_AVAILABLE}


@app.get("/api/seismic/status")
def seismic_status():
    return {"available": SEISMIC_AVAILABLE,
            "method": "Intercept time, two-layer & multilayer (Telford 1990)"}


# ── Saha Verisi Grid'leme ─────────────────────────────────────────────────────
class FieldGridRequest(BaseModel):
    channel_data: dict   # {"U_ppm": [v1,v2,...], "SP_mV": [...]}
    obs_x:        List[float]
    obs_y:        List[float]
    domain_m:     float = 480.0
    nx:           int   = 32
    ny:           int   = 32
    nz:           int   = 32
    method:       str   = "rbf"   # rbf | idw | kriging
    extend_3d:    bool  = True


@app.post("/api/field/interpolate")
def field_interpolate(req: FieldGridRequest):
    """Düzensiz saha verisi → düzenli grid (RBF/IDW/Kriging)."""
    if not GRIDDING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Grid'leme modülü yüklenemedi.")
    import numpy as np
    motor = FieldGriddingMotor()
    obs_values = {k: np.array(v) for k, v in req.channel_data.items()}
    result = motor.grid_surface(
        np.array(req.obs_x), np.array(req.obs_y),
        obs_values,
        domain_m=req.domain_m,
        nx=req.nx, ny=req.ny,
        method=req.method,
        extend_3d=req.extend_3d,
        nz=req.nz,
    )
    return result


@app.get("/api/field/status")
def field_status():
    return {"available": GRIDDING_AVAILABLE,
            "methods": ["rbf", "idw", "kriging"],
            "reference": "Broomhead & Lowe (1988), Matheron (1963)"}


# ── Jeouzamsal Dışa Aktarma ───────────────────────────────────────────────────
from fastapi.responses import Response as FastAPIResponse

class GeoExportRequest(BaseModel):
    dataset:      Optional[str] = None
    selected_index: int  = 0
    layer_name:   str   = "anomaly"
    format:       str   = "geojson"   # geojson | contour_geojson | geotiff | png
    n_contours:   int   = 8
    channel:      str   = "max_projection"  # max_projection | slice_z | composite


@app.post("/api/export/geospatial")
def export_geospatial(req: GeoExportRequest):
    """Anomali haritası → GeoTIFF / GeoJSON / Shapefile."""
    if not EXPORT_GEO_AVAILABLE:
        raise HTTPException(status_code=503, detail="Export modülü yüklenemedi.")
    import numpy as np
    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids = resample_to_forward(model_native)

    # 2D projeksiyon
    if req.channel == "max_projection":
        grid_2d = model_fwd.max(axis=2)
    elif req.channel == "slice_z":
        grid_2d = model_fwd[:, :, model_fwd.shape[2]//2]
    else:
        grid_2d = model_fwd.max(axis=2)

    motor = GeoExportMotor()
    result = motor.export_grid(
        grid_2d, req.layer_name, req.format,
        n_contours=req.n_contours,
    )

    if isinstance(result["data"], bytes):
        return FastAPIResponse(
            content=result["data"],
            media_type=result["content_type"],
            headers={"Content-Disposition": f'attachment; filename="{result["filename"]}"'},
        )
    return {"data": result["data"], "filename": result["filename"],
            "format": result["format"], "content_type": result["content_type"]}


@app.get("/api/export/status")
def export_status():
    return {
        "available": EXPORT_GEO_AVAILABLE,
        "formats": ["geojson", "contour_geojson", "geotiff", "png"],
        "crs": "EPSG:4326 (WGS84) / EPSG:32636 (UTM 36N)",
        "qgis_compatible": True,
    }


class PinnInferRequest(BaseModel):
    dataset: Optional[str] = None
    selected_index: int = 0
    dataset_grav_mag: Optional[str] = None
    threshold: float = Field(default=0.35, ge=0.0, le=1.0)


@app.get("/api/pinn/status")
def pinn_status_endpoint():
    """GeoUNet 3D U-Net checkpoint ve model durumu."""
    if not PINN_AVAILABLE:
        return {"available": False, "error": "engines.pinn_stub yüklenemedi."}
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    return _pinn_status(backend_dir=backend_dir)


@app.post("/api/pinn/infer")
def pinn_infer(req: PinnInferRequest):
    """GeoUNet ile hızlı 3D jeolojik model tahmini.
    Giriş: gz(21×21) + mag(21×21) — X_mag_grav dosyasından veya Y self-forward.
    Çıkış: (32,32,32) geometri + binary mask + istatistikler."""
    if not PINN_AVAILABLE:
        raise HTTPException(status_code=503, detail="pinn_stub modülü yüklenemedi.")
    backend_dir = os.path.dirname(os.path.abspath(__file__))

    gz_map = mag_map = None
    if req.dataset_grav_mag:
        gm_path = _resolve_dataset_path(req.dataset_grav_mag)
        if gm_path:
            try:
                gm_sample = _load_indexed_npy(gm_path, req.selected_index)
                if gm_sample.ndim == 3 and gm_sample.shape[-1] == 2:
                    mag_map = gm_sample[..., 0]
                    gz_map  = gm_sample[..., 1]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"X_mag_grav okunamadı: {e}")

    if gz_map is None or mag_map is None:
        model_native, _ = load_model_native(req.dataset, req.selected_index)
        model_fwd, grids = resample_to_forward(model_native)
        gz_map  = forward_grav(model_fwd, grids)
        mag_map = forward_mag(model_fwd, grids)

    try:
        result = _pinn_infer(gz_map=gz_map, mag_map=mag_map,
                             backend_dir=backend_dir, threshold=req.threshold)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GeoUNet inference hatası: {e}")

    return {
        "model_data":  result["model_data"],
        "mask_data":   result["mask_data"],
        "stats":       result["stats"],
        "checkpoint":  result["checkpoint"],
        "device":      result["device"],
        "obs_source":  "X_mag_grav dosyası" if req.dataset_grav_mag else "Y self-forward (sentetik)",
        "meta": {"grid_size": 32, "domain_m": DOMAIN_EXTENT, "threshold": req.threshold},
    }


@app.get("/api/fvm/status")
def fvm_status():
    return {
        "available": FVM_AVAILABLE,
        "engines": {
            "prism": {"name": "Analitik Prizma (Nagy / Bhattacharyya)",
                      "type": "closed-form", "gpu": True,
                      "desc": "Sonsuz homojen uzay Green fonksiyonu — hızlı, analitik"},
            "fvm":   {"name": "Sonlu Hacimler (Poisson, bounded domain)",
                      "type": "numerical", "gpu": False,
                      "desc": "∇²U = kaynak, Dirichlet BC — sınırlı domain, daha gerçekçi sınır koşulları"},
        }
    }


class FVMCompareRequest(BaseModel):
    dataset: Optional[str] = None
    selected_index: int = 0
    grav_active: bool = True
    mag_active: bool = True


@app.post("/api/fvm/compare")
def fvm_compare(req: FVMCompareRequest):
    """Prizma (analitik) ve FVM (nümerik) motorlarını aynı model üzerinde çalıştırır,
    sonuçları ve farkı (RMSE, maksimum sapma) döndürür."""
    if not FVM_AVAILABLE:
        raise HTTPException(status_code=503,
            detail="FVM modülleri yüklenemedi. engines/gravity_fvm.py ve magnetic_fvm.py kontrol edin.")

    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids = resample_to_forward(model_native)

    result = {}

    if req.grav_active:
        import time
        t0 = time.perf_counter()
        gz_prism = forward_grav(model_fwd, grids, engine_mode="prism")
        t_prism = time.perf_counter() - t0

        t0 = time.perf_counter()
        gz_fvm   = forward_grav(model_fwd, grids, engine_mode="fvm")
        t_fvm = time.perf_counter() - t0

        diff = gz_prism - gz_fvm
        result["gravity"] = {
            "prism":      gz_prism.tolist(),
            "fvm":        gz_fvm.tolist(),
            "diff":       diff.tolist(),
            "rmse_mgal":  float(np.sqrt(np.mean(diff**2))),
            "max_diff_mgal": float(np.max(np.abs(diff))),
            "rel_rmse_pct": float(np.sqrt(np.mean(diff**2)) / (np.std(gz_prism)+1e-12) * 100),
            "time_prism_s": round(t_prism, 4),
            "time_fvm_s":   round(t_fvm, 4),
        }

    if req.mag_active:
        import time
        t0 = time.perf_counter()
        dt_prism = forward_mag(model_fwd, grids, engine_mode="prism")
        t_prism = time.perf_counter() - t0

        t0 = time.perf_counter()
        dt_fvm   = forward_mag(model_fwd, grids, engine_mode="fvm")
        t_fvm = time.perf_counter() - t0

        diff = dt_prism - dt_fvm
        result["magnetic"] = {
            "prism":      dt_prism.tolist(),
            "fvm":        dt_fvm.tolist(),
            "diff":       diff.tolist(),
            "rmse_nt":    float(np.sqrt(np.mean(diff**2))),
            "max_diff_nt": float(np.max(np.abs(diff))),
            "rel_rmse_pct": float(np.sqrt(np.mean(diff**2)) / (np.std(dt_prism)+1e-12) * 100),
            "time_prism_s": round(t_prism, 4),
            "time_fvm_s":   round(t_fvm, 4),
        }

    return {
        "result": result,
        "dataset_used": os.path.basename(used_path) if used_path else "demo_sentetik",
        "grid_size": NBC_FORWARD,
        "note": "FVM sınırlı domain (Dirichlet BC), Prizma sonsuz uzay (Green fn)."
    }


# ── PyInstaller / Electron giriş noktası ──────────────────────────────────────
# `uvicorn server:app --reload ...` ile dıştan çalıştırmanın yanı sıra,
# PyInstaller ile derlenmiş server.exe'nin kendi kendine ayağa kalkabilmesi
# için (bkz. PAKETLEME_REHBERI.md) bu blok gerekli.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



# ── SimPEG Entegrasyonu ───────────────────────────────────────────────────────
try:
    import discretize
    from simpeg import maps, data_misfit, regularization, optimization
    from simpeg import inversion, inverse_problem, data as simpeg_data
    from simpeg.potential_fields import gravity as simpeg_grav
    from simpeg.potential_fields import magnetics as simpeg_mag
    from simpeg.directives import (
        BetaSchedule, TargetMisfit, BetaEstimate_ByEig, UpdateSensitivityWeights
    )
    SIMPEG_AVAILABLE = True
except ImportError:
    SIMPEG_AVAILABLE = False


class SimPEGInversionRequest(BaseModel):
    grav_active:    bool  = True
    mag_active:     bool  = True
    dataset:        Optional[str] = None
    dataset_grav_mag: Optional[str] = None
    selected_index: int   = 0
    nbc:            int   = Field(default=16, ge=8, le=32)
    max_iter:       int   = Field(default=15, ge=5, le=50)
    alpha_s:        float = Field(default=1e-4, ge=1e-6, le=1.0)   # smallness
    alpha_x:        float = Field(default=1.0,  ge=0.0, le=10.0)   # smoothness
    chifact:        float = Field(default=1.0,  ge=0.5, le=5.0)    # misfit hedefi
    noise_floor_grav: float = 0.005   # mGal
    noise_floor_mag:  float = 0.5     # nT
    seed:           int   = 42


@app.get("/api/simpeg/status")
def simpeg_status():
    return {"available": SIMPEG_AVAILABLE,
            "version": __import__('simpeg').__version__ if SIMPEG_AVAILABLE else None}


@app.post("/api/simpeg/inversion")
def simpeg_inversion(req: SimPEGInversionRequest):
    """
    SimPEG Tikhonov L2 inversion — adjoint tabanlı, sensitivity matrix ile.

    Kendi Adam/gradient çözücümüzden farkı:
      - Sensitivity matrix (G) analitik olarak hesaplanır → gerçek gradyan
      - Gauss-Newton iterasyonu → kuadratik yakınsama
      - BetaSchedule + TargetMisfit → otomatik regularizasyon ayarı
      - Sensitivity weighting → derinliğe bağlı çözünürlük dengeleme

    Çıktı: density (g/cc) ve susceptibility (SI) modelleri ayrı ayrı.
    """
    if not SIMPEG_AVAILABLE:
        raise HTTPException(status_code=503,
            detail="SimPEG kurulu değil. 'pip install simpeg discretize' çalıştırın.")

    if not req.grav_active and not req.mag_active:
        raise HTTPException(status_code=400, detail="En az bir yöntem aktif olmalı.")

    nbc = req.nbc
    dh  = DOMAIN_EXTENT / nbc

    # Mesh — GeoPINN koordinat sistemi ile tutarlı:
    # X: -half..+half (doğu-batı)
    # Y: -half..+half (kuzey-güney)  
    # Z: -DOMAIN_EXTENT..0 (derinlik, yüzey=0, taban=-480m)
    half = DOMAIN_EXTENT / 2
    mesh = discretize.TensorMesh(
        [np.ones(nbc)*dh, np.ones(nbc)*dh, np.ones(nbc)*dh],
        origin=[-half, -half, -DOMAIN_EXTENT]
    )

    # Gözlem noktaları — yüzey z=0, GeoPINN 21×21 ızgarasıyla aynı
    obs_1d = np.linspace(-half, half, 21)
    ox, oy = np.meshgrid(obs_1d, obs_1d)
    # Z=0: yüzey (SimPEG gravity receiver yüzey üstünde olmalı, +1m)
    obs_pts = np.c_[ox.ravel(), oy.ravel(), np.full(ox.size, 0.5)]

    # d_obs yükle (X_mag_grav varsa oradan, yoksa Y'den forward)
    model_native, used_path = load_model_native(req.dataset, req.selected_index)
    model_fwd, grids        = resample_to_forward(model_native, nbc=nbc)

    d_obs_g = d_obs_m = None
    gm_path = _resolve_dataset_path(req.dataset_grav_mag) if req.dataset_grav_mag else None
    if gm_path:
        gm = _load_indexed_npy(gm_path, req.selected_index)
        if gm.ndim == 3 and gm.shape[-1] == 2:
            if req.mag_active:  d_obs_m = gm[..., 0].ravel()
            if req.grav_active: d_obs_g = gm[..., 1].ravel()

    # Yoksa kendi motorumuzla forward
    if req.grav_active and d_obs_g is None:
        d_obs_g = forward_grav(model_fwd, grids).ravel()
    if req.mag_active and d_obs_m is None:
        d_obs_m = forward_mag(model_fwd, grids).ravel()

    # SimPEG birime çevir: bizim gz mGal (zaten), mag nT (zaten)
    # SimPEG gravity input density: g/cc; mag susceptibility: SI

    reg_kw = dict(alpha_s=req.alpha_s, alpha_x=req.alpha_x,
                  alpha_y=req.alpha_x, alpha_z=req.alpha_x)
    directives_fn = lambda: [
        UpdateSensitivityWeights(),
        BetaEstimate_ByEig(beta0_ratio=10),
        TargetMisfit(chifact=req.chifact),
        BetaSchedule(coolingFactor=2, coolingRate=2),
    ]

    results = {}
    history = {"grav": [], "mag": []}

    # ── Gravity inversion ──────────────────────────────────────────────────────
    if req.grav_active and d_obs_g is not None:
        rx_g   = simpeg_grav.receivers.Point(obs_pts, components=["gz"])
        src_g  = simpeg_grav.sources.SourceField([rx_g])
        surv_g = simpeg_grav.survey.Survey(src_g)
        sim_g  = simpeg_grav.simulation.Simulation3DIntegral(
            mesh, survey=surv_g,
            rhoMap=maps.IdentityMap(mesh),
            store_sensitivities="ram",
        )
        # İşaret: SimPEG gz yukarı-pozitif, bizim d_obs aşağı-pozitif
        dat_g  = simpeg_data.Data(surv_g,
                    dobs=d_obs_g,        # SimPEG gz: aşağı-pozitif (GeoPINN ile aynı)
                    relative_error=0.02,
                    noise_floor=req.noise_floor_grav)
        dmis_g = data_misfit.L2DataMisfit(simulation=sim_g, data=dat_g)
        reg_g  = regularization.WeightedLeastSquares(mesh, **reg_kw)
        opt_g  = optimization.InexactGaussNewton(maxIter=req.max_iter, cg_maxiter=20)
        prob_g = inverse_problem.BaseInvProblem(dmis_g, reg_g, opt_g)

        # Iter callback → history topla
        iter_log_g = []
        class _LogCB:
            def __init__(self, log): self.log = log
            def call(self, xc, f, proj_xc): self.log.append(float(f))
        cb_g = _LogCB(iter_log_g)
        opt_g.callback = cb_g.call

        inv_g = inversion.BaseInversion(prob_g, directiveList=directives_fn())
        m_grav = inv_g.run(1e-5 * np.ones(mesh.nC))
        results["density_gcc"]    = m_grav.tolist()
        results["density_max"]    = float(m_grav.max())
        results["density_kgm3"]   = float(m_grav.max() * 1000)
        history["grav"]           = iter_log_g

    # ── Magnetics inversion ────────────────────────────────────────────────────
    if req.mag_active and d_obs_m is not None:
        rx_m   = simpeg_mag.receivers.Point(obs_pts, components=["tmi"])
        src_m  = simpeg_mag.sources.UniformBackgroundField(
            [rx_m], amplitude=47000, inclination=60, declination=5)
        surv_m = simpeg_mag.survey.Survey(src_m)
        sim_m  = simpeg_mag.simulation.Simulation3DIntegral(
            mesh, survey=surv_m,
            chiMap=maps.IdentityMap(mesh),
            store_sensitivities="ram",
        )
        dat_m  = simpeg_data.Data(surv_m,
                    dobs=d_obs_m,
                    relative_error=0.02,
                    noise_floor=req.noise_floor_mag)
        dmis_m = data_misfit.L2DataMisfit(simulation=sim_m, data=dat_m)
        reg_m  = regularization.WeightedLeastSquares(mesh, **reg_kw)
        opt_m  = optimization.InexactGaussNewton(maxIter=req.max_iter, cg_maxiter=20)
        prob_m = inverse_problem.BaseInvProblem(dmis_m, reg_m, opt_m)

        iter_log_m = []
        class _LogM:
            def __init__(self, log): self.log = log
            def call(self, xc, f, proj_xc): self.log.append(float(f))
        cb_m = _LogM(iter_log_m)
        opt_m.callback = cb_m.call

        inv_m = inversion.BaseInversion(prob_m, directiveList=directives_fn())
        m_mag = inv_m.run(1e-6 * np.ones(mesh.nC))
        results["susceptibility_SI"]  = m_mag.tolist()
        results["susceptibility_max"] = float(m_mag.max())
        history["mag"]                = iter_log_m

    # Birleşik model: normalize density + susceptibility ortalaması
    if "density_gcc" in results and "susceptibility_SI" in results:
        rho_n = np.array(results["density_gcc"])
        chi_n = np.array(results["susceptibility_SI"])
        rho_n = (rho_n - rho_n.min()) / (rho_n.max() - rho_n.min() + 1e-10)
        chi_n = (chi_n - chi_n.min()) / (chi_n.max() - chi_n.min() + 1e-10)
        combined = (rho_n + chi_n) / 2
        results["combined_model"] = combined.reshape(nbc, nbc, nbc).tolist()
    elif "density_gcc" in results:
        arr = np.array(results["density_gcc"])
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        results["combined_model"] = arr.reshape(nbc, nbc, nbc).tolist()
    elif "susceptibility_SI" in results:
        arr = np.array(results["susceptibility_SI"])
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        results["combined_model"] = arr.reshape(nbc, nbc, nbc).tolist()

    return {
        "model_data"   : results.get("combined_model"),
        "results"      : {k: v for k, v in results.items() if k != "combined_model"
                          and not isinstance(v, list)},
        "history"      : history,
        "mesh_info"    : {"nbc": nbc, "dh_m": dh, "domain_m": DOMAIN_EXTENT,
                          "n_cells": mesh.nC},
        "dataset_used" : os.path.basename(used_path) if used_path else "demo_sentetik",
        "method"       : "SimPEG Tikhonov L2 (adjoint-based, Gauss-Newton)",
    }

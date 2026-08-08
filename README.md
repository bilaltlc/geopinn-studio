<div align="center">

```
██████╗ ███████╗ ██████╗ ██████╗ ██╗███╗   ██╗███╗   ██╗
██╔════╝ ██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║████╗  ██║
██║  ███╗█████╗  ██║   ██║██████╔╝██║██╔██╗ ██║██╔██╗ ██║
██║   ██║██╔══╝  ██║   ██║██╔═══╝ ██║██║╚██╗██║██║╚██╗██║
╚██████╔╝███████╗╚██████╔╝██║     ██║██║ ╚████║██║ ╚████║
 ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝
```

**STUDIO 3.0** — Applied Geophysics Suite

*Gravity · Magnetics · CSAMT · Radiometry · Heat Flow · FVM*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org)
[![SimPEG](https://img.shields.io/badge/SimPEG-0.25.2-green.svg)](https://simpeg.xyz)
[![Electron](https://img.shields.io/badge/Electron-33-47848F.svg)](https://electronjs.org)

</div>

---

## Overview

GeoPINN Studio is a desktop application for 3D geophysical forward modelling, joint inversion, and mineral exploration analysis. Built for REE-F-Ba-Th deposit exploration (Beylikova analogue), it integrates multiple geophysical methods in a unified GPU-accelerated environment.

**Architecture:** React + Three.js frontend · FastAPI backend · PyTorch GPU engines · Colab/local deployment

---

## v3.0.0.1 — What's New

| Feature | Details |
|---------|---------|
| **Radiometry & Heat Flow** | U/Th/K → gammaray forward model, REE alteration index, FVM heat conduction |
| **FVM Engine** | Poisson bounded domain solver, prism vs FVM comparison |
| **Data Format Selector** | 7 geophysical formats auto-detected (CSV/XYZ/DAT) |
| **Dual Theme** | Full dark `#0A0C0F` ↔ full light, persistent |
| **Vertical Tab Rail** | 8 right-panel tabs in 46px icon rail |
| **System Tray** | Minimize to tray, maximize on launch (Electron) |

---

## Screenshots

### Split View — İstatistik + 3D Viewer
![Split Stats](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_split_stats.png)
*İST butonu — istatistik paneli 3D viewer yanında, aynı React state paylaşıyor*

### Split View — Leaflet Harita + 3D
![Split Map](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_split_map.png)
*HRT butonu — Beylikova REE anomalisi OSM üzerinde, 3D model yanında*

### Split View — Kesit + 3D
![Split Slice](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_split_slice.png)
*KST butonu — Z derinlik kesiti ve 3D izoyüzey eş zamanlı*

### Leaflet Anomali Haritası — OSM Üzerinde
![Leaflet Map](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_leaflet_map.png)
*Beylikova REE alanı · OSM tile + ImageOverlay · mavi→yeşil→sarı→kırmızı skalası*

### Kesit Görünümü (Z ekseni)
![Slice Z](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_leaflet_slice2.png)
*Z=5 derinlik kesiti · viridis renk skalası · 16³ grid*

### IP — Cole-Cole Pseudosection
![IP Panel](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_ip_panel.png)
*Cole-Cole parametreleri · Chargeability maks: 0.052 · Faz: 17.6 mrad · Orta sülfür alterasyonu*

### SP — Öz-Potansiyel Anomali
![SP Panel](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_sp_panel.png)
*Elektrokinetik baskın kaynak · SP min: -390 mV · Güçlü hidrotermal sistem*

### Füzyon — 7 Yöntem Bileşik Anomali
![Fusion Panel](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_fusion_panel.png)
*Fuzzy Gamma γ=0.85 · Bileşik maks: 1.000 · Bölge 1 skor: 1.000 · 4 aktif yöntem*

### Rehber — IP & SP Sekmeleri
![Guide IP SP](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_guide_ip_sp.png)
*Pelton (1978), Revil & Leroy (2004), Sill (1983) referanslı jeolojik yorum kılavuzu*

### SimPEG Tikhonov Inversion Sonuçları
![SimPEG Results](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_simpeg_results.png)
*Δρ: 10.9 kg/m³ · χ: 7.09e-4 SI · Gravite + Manyetik split section*

### Joint Inversion — Ters Çözüm
![Joint Inversion](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_joint_inversion.png)
*Misfit: 100.279 → 0.2405 · RMSE: 0.0636 · 16³ grid · 150 iter*

### Belirsizlik Analizi (UQ)
![UQ Panel](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_uq_panel.png)
*5 realizasyon · %3 gürültü · Yüksek güven: %100 · RMSE: 0.0643*

### Radyometri & Isı Akışı
![Radiometry](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_3d_radiometry2.png)
*U/Th/K → Th/U: 4.00 · TC: 56.8 cps · Ort. ısı akışı: -140629 mW/m²*

---

## Physics Engines

### Forward Modelling
| Engine | Method | Backend | GPU |
|--------|--------|---------|-----|
| `gravity_prism.py` | Nagy analytic | PyTorch | ✓ |
| `magnetic_prism.py` | Bhattacharyya TMI | PyTorch | ✓ |
| `csamt_1d.py` | Wait recursion MT | PyTorch | ✓ |
| `gravity_fvm.py` | Poisson FVM (∇²U=4πGρ) | scipy.sparse | — |
| `magnetic_fvm.py` | Poisson FVM (∇²φ=∇·M) | scipy.sparse | — |
| `heat_flow_fvm.py` | Heat conduction (∇²T=-Q/k) | scipy.sparse | — |
| `radiometry.py` | U/Th/K → gammaray forward | numpy | — |
| `ip_forward.py` | Cole-Cole IP, dipole-dipole | PyTorch | ✓ |
| `sp_forward.py` | Electrokinetic + thermoelectric SP | numpy | — |
| `data_fusion.py` | 7-method composite anomaly | numpy/scipy | — |

### Inversion
| Method | Algorithm | Notes |
|--------|----------|-------|
| Joint Inversion | Adam + autograd | Grav + Mag + CSAMT simultaneous |
| SimPEG Tikhonov | Gauss-Newton + adjoint | L2 regularization, sensitivity matrix |
| Uncertainty (UQ) | Monte Carlo | n_realizations, CV/P10/P50/P90 |

### Petrophyics (Beylikova REE Analogue)
```
ρ(x) = 2.70 + 2.00 × f(x)      [g/cm³]   density
χ(x) = 1e-4 + 3e-4 × f(x)      [SI]      susceptibility  
ρₑ(x) = 500 × 0.10^f(x)        [Ω·m]     resistivity
Q(x)  = ρ·(9.52e-5·U + 2.56e-5·Th + 3.48e-6·K)  [W/m³]  heat production
```

---

## Workflow — Beylikova REE Exploration

```
① Geometry Generator
   └─ Beylikova Vein type · dip=60° · depth=30-400m · width=60m
        ↓
② Forward Modelling  
   └─ Gravity + Magnetics (GPU) · engine: Prism or FVM
        ↓
③ Radiometry & Heat Flow
   └─ U/Th/K → Th/U ratio → REE alteration index → surface heat flux
        ↓
④ Joint Inversion
   └─ Adam optimizer · 150 iter · 16³-64³ grid · Grav+Mag+CSAMT
        ↓
⑤ Uncertainty Analysis  
   └─ 8 realizations · 5% noise · CV map · high-confidence zones
        ↓
⑥ FVM Validation
   └─ Prism vs FVM RMSE · boundary effect check
        ↓
⑦ Export
   └─ CSV (model points) · PNG (anomaly maps)
```

---

## FVM Results (v3.0.0.1)

Benchmark on `Y_beylikova_vein_32x32x32.npy` (32³ grid):

| Method | RMSE | Max Diff | Rel. RMSE | Time |
|--------|------|----------|-----------|------|
| **Gravity** (mGal) | 0.0126 | 0.0257 | 5.29% | Prism: 0.148s · FVM: 7.245s |
| **Magnetics** (nT) | 1.6218 | 7.5073 | 101.62% | Prism: 0.211s · FVM: 0.686s |

> Magnetic FVM divergence (101%) is expected — FVM uses scalar potential approximation (∇²φ=∇·M) vs Bhattacharyya full tensor. Domain boundary effects dominate for thin vein geometries. Gravity FVM at 5.29% is within acceptable range.

---

## Installation

### Prerequisites
```bash
# Backend
pip install fastapi uvicorn numpy scipy torch pyngrok
pip install simpeg discretize          # optional — SimPEG inversion
pip install harmonica verde            # optional — validation

# Frontend
npm install
```

### Run (Development)
```bash
# 1. Start backend (local or Colab)
cd geopinn-backend
uvicorn server:app --host 127.0.0.1 --port 8000

# 2. Start frontend
cd geopinn-frontend
npm run dev
```

### Run on Google Colab (GPU)
```python
# In Colab notebook:
!pip install fastapi uvicorn pyngrok simpeg discretize
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
# Run Server Başlat cell → copy ngrok URL → paste in app Settings
```

### Build Desktop App (Windows)
```bash
cd geopinn-frontend
npm run dist
# Output: dist-electron/GeoPINN Studio-win.zip
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server status |
| `/api/run-physics-engine` | POST | Forward modelling (prism/fvm) |
| `/api/joint-inversion` | POST | Adam joint inversion |
| `/api/uncertainty` | POST | Monte Carlo UQ |
| `/api/simpeg/inversion` | POST | SimPEG Tikhonov L2 |
| `/api/fvm/compare` | POST | Prism vs FVM benchmark |
| `/api/radiometry/forward` | POST | U/Th/K → gammaray + heat flow |
| `/api/generate-geometry` | POST | Synthetic geometry generator |
| `/api/analyses` | GET/POST/DELETE | Analysis history |
| `/api/data/list` | GET | Dataset management |

Full API docs: `http://localhost:8000/docs`

---

## Data Formats

| Format Key | Extensions | Columns |
|-----------|-----------|---------|
| `model_npy` | `.npy` | GeoPINN 3D grid |
| `radiometry_csv` | `.csv .xyz .dat` | x, y, U_ppm, Th_ppm, K_pct |
| `sp_csv` | `.csv .dat` | x, y, SP_mV |
| `ip_csv` | `.csv .dat` | x_mid, depth, chargeability_ms, resistivity_ohmm |
| `seismic_refrac_csv` | `.csv .dat` | offset_m, tt_ms |
| `grav_csv` | `.csv .dat` | x, y, gz_mGal |
| `mag_csv` | `.csv .dat` | x, y, TMI_nT |

---

## Roadmap

### v3.1.0
- [ ] `data_fusion.py` — multi-method composite anomaly score (Co-kriging)
- [ ] IP forward engine — Cole-Cole complex resistivity
- [ ] SP forward engine — electrokinetic coupling (∇²V = ∇·(L·∇T))
- [ ] Field data gridding — irregular → regular (RBF interpolation)

### v4.0.0 — PINN Integration
- [ ] Physics-Informed Neural Networks — PDE loss (Laplace/Poisson)
- [ ] PyTorch autograd physics layer (`physics_eqs.py`)
- [ ] Multi-method PINN joint inversion
- [ ] Transfer learning: synthetic → real field data

---

## Citation

```bibtex
@software{geopinn_studio_2026,
  title   = {GeoPINN Studio: Applied Geophysics Desktop Suite},
  version = {3.0.0.1},
  year    = {2026},
  url     = {https://github.com/bilaltlc/geopinn-studio},
  note    = {Beylikova REE-F-Ba-Th exploration analogue}
}
```

---


---

## Known Limitations & Future Work

### 1. Hyperparameter Sensitivity
The regularization weight `reg_lambda` and per-method weights (gravity/magnetics/CSAMT) require manual tuning. Small increases in `reg_lambda` can degrade misfit and RMSE rapidly, indicating sensitivity to loss term balance.

**Planned fix (v3.1):** Adaptive Loss Balancing (GradNorm or uncertainty weighting) — automatic per-method weight adjustment during joint inversion.

### 2. Computational Cost at High Resolution
At `32³` or higher grid resolution, inversion time increases significantly. GPU memory management (PyTorch CUDA) must be stable end-to-end for large field datasets. FVM solver (scipy.sparse CG) is CPU-only and scales poorly beyond `32³`.

**Planned fix (v3.1):** GPU-accelerated sparse solver option; adaptive grid refinement (coarse → fine).

### 3. Geological Validation Gap
Current mass (20.7 Mt) and volume (36.5 Mm³) estimates are mathematically convergent but have not been cross-validated against:
- Known geological cross-sections of the Beylikova area
- Drill-hole data (lithology, assay)
- Published REE deposit analogues (Bayan Obo, Mountain Pass)

**For academic use:** Results should be treated as forward-model-consistent estimates, not ground-truth reserves. Systematic validation against borehole data is required before any resource classification.

### 4. 1D CSAMT Assumption
The CSAMT engine (`csamt_1d.py`) uses a 1D layered earth model (Wait recursion). Real CSAMT data in structurally complex terrains (fault zones, steeply dipping veins) will show 2D/3D effects not captured by 1D inversion.

**Planned fix (v4.0):** 2.5D CSAMT forward operator integrated with PINN framework.

---

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">
<sub>GeoPINN Studio · Applied Geophysics · telcihamdibilal@gmail.com</sub>
</div>

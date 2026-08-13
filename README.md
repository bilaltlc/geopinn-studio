<div align="center">

```
██████╗ ███████╗ ██████╗ ██████╗ ██╗███╗   ██╗███╗   ██╗
██╔════╝ ██╔════╝██╔═══██╗██╔══██╗██║████╗  ██║████╗  ██║
██║  ███╗█████╗  ██║   ██║██████╔╝██║██╔██╗ ██║██╔██╗ ██║
██║   ██║██╔══╝  ██║   ██║██╔═══╝ ██║██║╚██╗██║██║╚██╗██║
╚██████╔╝███████╗╚██████╔╝██║     ██║██║ ╚████║██║ ╚████║
 ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝
```

**v4.0 BETA** — Applied Geophysics Suite

*Gravity · Magnetics · CSAMT · IP · SP · Radiometry · Heat Flow · FVM · GeoUNet*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org)
[![SimPEG](https://img.shields.io/badge/SimPEG-0.25.2-green.svg)](https://simpeg.xyz)
[![Electron](https://img.shields.io/badge/Electron-33-47848F.svg)](https://electronjs.org)

</div>

---

## Overview

GeoPINN Studio is a desktop application for 3D geophysical forward modelling, joint inversion, and mineral exploration analysis — now with **GeoUNet learned inversion** (surrogate neural network). Built for REE-F-Ba-Th deposit exploration (Beylikova analogue), it integrates multiple geophysical methods in a unified GPU-accelerated environment.

**Architecture:** React + Three.js frontend · FastAPI backend · PyTorch GPU engines · Colab/local deployment

---

## v4.0 BETA — What's New

| Feature | Details |
|---------|---------|
| **GeoUNet PINN** | ObsEncoder-2DCNN + 3D U-Net · IoU=0.57 · ~0.03s GPU inference |
| **Learned Inversion** | gz+mag anomaly maps → 3D ore geometry · non-iterative |
| **3D Popup Modal** | 82vw×82vh floating viewer — no longer fixed center |
| **Center Grid** | 2×2 nav cards: 3D Model / İstatistik / Kesit / Anomali Haritası |
| **Panel Resize** | Left/right panels draggable (200–560px range) |
| **Dark Mode Fix** | ColorBar, overlays, radyometri panel fully theme-aware |
| **Slice Color Fix** | P2–P98 percentile normalization — no more white saturation |
| **Workflow** | 8-step guide including GeoUNet and Binary Mask steps |
| **Colab Notebook** | Drive-based, no GitHub clone required |

---

## Screenshots

### v4.0 — Ana Ekran (2×2 Navigasyon Grid)
![Ana Ekran](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_ana_ekran.png)
*3D Model / İstatistik / Kesit / Anomali Haritası kartları · "Görüntülemek için önce bir analiz çalıştırın"*

### v4.0 — GeoUNet 3D Popup (Y→Forward, İki Cevher Gövdesi)
![3D Popup Y Forward](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_3d_popup_geounet_y_forward.png)
*Y_beylikova_vein_32x32x32 · Y→Forward mod · İki ayrı cevher gövdesi · %6.7 fraksiyon · 7.46 Mm³*

### v4.0 — GeoUNet 3D Popup (X Dosyası, Damar Geometrisi)
![3D Popup X](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_3d_popup_geounet_x_dosyasi.png)
*X_mag_grav.npy · Gerçek anomali verisi · Damar şekli · Cyan-mor renk skalası · 12.50 Mm³*

### v4.0 — Anomali Haritası + PINN Paneli (X Dosyası)
![Anomali Harita X PINN](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_anomali_harita_x_dosyasi_pinn.png)
*HRT modu · Checkpoint yüklü (22.2 MB, CUDA) · X_mag_grav.npy overlay · Beylikova 39°55N 31°40E*

### v4.0 — Anomali Haritası (Y→Forward)
![Anomali Harita Y](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_anomali_harita_y_forward.png)
*HRT modu · Y_beylikova_vein_32x32x32 · İki yüksek anomali bölgesi · Y→Forward mod*

### v4.0 — Binary Mask Anomali Haritası (X Dosyası)
![Binary Mask Map](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_anomali_harita_binary_mask.png)
*HRT modu · Binary Mask → Harita · Threshold=0.35 üstü kırmızı · %11.3 fraksiyon · 12.50 Mm³*

### v4.0 — Kesit Görünümü (Y→Forward, İki Gövde)
![Kesit Y](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_kesit_y_forward_iki_govde.png)
*KST modu · Z kesiti · İki ayrı cevher bloğu · Viridis · %6.7 fraksiyon*

### v4.0 — Kesit Görünümü (X Dosyası)
![Kesit X](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_kesit_geounet_x_dosyasi.png)
*KST modu · X_mag_grav · Dairesel yüksek anomali · %11.3 fraksiyon · 12.50 Mm³*

### v4.0 — İstatistik Paneli (Y→Forward)
![İstatistik Y](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_istatistik_y_forward.png)
*İST modu · Ort: 0.066 · P90: 0.079 · >0.5 hücre: 1925/32768 · %6.7 fraksiyon*

### v4.0 — İstatistik Paneli (X Dosyası)
![İstatistik X](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/v4_istatistik_x_dosyasi_pinn.png)
*İST modu · X_mag_grav · Ort: 0.109 · P90: 0.632 · >0.5 hücre: 3480/32768 · %11.3 fraksiyon*

### IP — Cole-Cole Pseudosection
![IP Panel](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_ip_panel.png)
*Cole-Cole parametreleri · Chargeability maks: 0.052 · Faz: 17.6 mrad · Orta sülfür alterasyonu*

### SP — Öz-Potansiyel Anomali
![SP Panel](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_sp_panel.png)
*Elektrokinetik baskın kaynak · SP min: -390 mV · Güçlü hidrotermal sistem*

### Füzyon — 7 Yöntem Bileşik Anomali
![Fusion Panel](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_fusion_panel.png)
*Fuzzy Gamma γ=0.85 · Bileşik maks: 1.000 · 4 aktif yöntem*

### Joint Inversion — Ters Çözüm
![Joint Inversion](https://raw.githubusercontent.com/bilaltlc/geopinn-studio/main/docs/screenshots/ss_joint_inversion.png)
*Misfit: 100.279 → 0.2405 · RMSE: 0.0636 · 16³ grid · Adam + autograd*

---

## GeoUNet — Learned Inversion

**Architecture**
```
Input: gz(21×21 mGal) + mag(21×21 nT)  →  resize 32×32
ObsEncoder:  Conv2d(2→32→64) + Conv2d(64, 64×g, 1) → view(B,64,32,32,32)
Encoder:     e1(64→32) → e2(32→64) → e3(64→128)  [MaxPool3d×3]
Bottleneck:  bot(128→256)
Decoder:     u3+d3 → u2+d2 → u1+d1  [ConvTranspose3d + skip concat]
Output:      Conv3d(32→1) + Sigmoid  →  (B, 32, 32, 32)
Norm:        GroupNorm(min(8,ch)) + GELU
```

**Training**
```
Data:       make_vein() synthetic vein geometry (32³ grid, seed=42)
Loss:       MSE + 0.5×Dice + 0.01×Laplacian + 0.005×TV
Optimizer:  AdamW · CosineAnnealingLR · grad_clip=1.0
Epochs:     204  · val_iou=0.5703  · val_mae=0.0221
Platform:   Kaggle T4 GPU
```

**Normalization (must match training)**
```
gz  / 5.97×10⁻⁷   (training std, grav_fwd output)
mag / 6.73×10⁻²   (training std, mag_fwd output)
```

**Industry terminology:** Surrogate inversion network / learned geophysical inversion / non-iterative forward-trained inverse model.
Ref: Wu & McMechan (2019), Sun & Demanet (2020)

---

## Physics Engines

### Forward Modelling
| Engine | Method | Backend | GPU |
|--------|--------|---------|-----|
| `gravity_prism.py` | Nagy analytic kernel | PyTorch | ✓ |
| `magnetic_prism.py` | Bhattacharyya TMI | PyTorch | ✓ |
| `csamt_1d.py` | Wait recursion MT | PyTorch | ✓ |
| `gravity_fvm.py` | Poisson FVM (∇²U=4πGρ) | scipy.sparse | — |
| `magnetic_fvm.py` | Poisson FVM (∇²φ=∇·M) | scipy.sparse | — |
| `heat_flow_fvm.py` | Heat conduction (∇²T=-Q/k) | scipy.sparse | — |
| `radiometry.py` | U/Th/K → gammaray forward | numpy | — |
| `ip_forward.py` | Cole-Cole IP, dipole-dipole | PyTorch | ✓ |
| `sp_forward.py` | Electrokinetic + thermoelectric SP | numpy | — |
| `data_fusion.py` | 7-method composite anomaly | numpy/scipy | — |
| `pinn_stub.py` | **GeoUNet 3D U-Net inference** | PyTorch | ✓ |

### Inversion
| Method | Algorithm | Notes |
|--------|----------|-------|
| **GeoUNet** | 3D U-Net (learned) | gz+mag → 3D model · ~0.03s GPU |
| Joint Inversion | Adam + autograd | Grav + Mag + CSAMT simultaneous |
| SimPEG Tikhonov | Gauss-Newton + adjoint | L2 regularization |
| Uncertainty (UQ) | Monte Carlo | n_realizations, CV/P10/P50/P90 |

### Petrophysics (Beylikova REE Analogue)
```
ρ(x)  = 2.70 + 2.00 × f(x)      [g/cm³]   density
χ(x)  = 1e-4 + 3e-4 × f(x)      [SI]      susceptibility
ρₑ(x) = 500 × 0.10^f(x)         [Ω·m]     resistivity
Q(x)  = ρ·(9.52e-5·U + 2.56e-5·Th + 3.48e-6·K)  [W/m³]  heat production
```

---

## Workflow — v4.0 Beylikova REE Exploration

```
① Load Data / Generate Geometry
   └─ .npy upload (Y model cube / X anomaly maps) or Geometry Generator
        ↓
② Forward Modelling
   └─ Gravity + Magnetics (GPU) · Prism analytic or FVM
        ↓
③ GeoUNet Inference  ◄─── NEW in v4.0
   └─ Y→Forward or X file · threshold 0.35-0.45 · ~0.03s · 32³ output
        ↓
④ IP / SP / Radiometry
   └─ Cole-Cole chargeability · SP electrokinetic · Th/U REE index
        ↓
⑤ Joint Inversion
   └─ Adam optimizer · 60 iter · 32³ grid · Grav+Mag+CSAMT
        ↓
⑥ Fusion + Uncertainty
   └─ Fuzzy Gamma 7-method · CV map · high-confidence zones
        ↓
⑦ 3D Visualize (Popup)
   └─ Isosurface · Binary Mask · opacity/threshold sliders
        ↓
⑧ Export
   └─ CSV (model points) · PNG (anomaly maps)
```

---

## Installation

### Option 1 — Electron Desktop (Local)
```bash
# Backend
cd geopinn-backend
pip install fastapi uvicorn numpy scipy torch python-multipart
uvicorn server:app --host 127.0.0.1 --port 8000

# Frontend
cd geopinn-frontend
npm install
npm run dev        # development
npm run dist       # package .exe
```

### Option 2 — Google Colab GPU
1. Open `geopinn-backend/GeoPINN_Studio_Colab.ipynb` in Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run cells 1–6 in order
4. Cell 5: upload `server.py` + all `engines/*.py` files
5. Cell 6: upload `geopinn_unet_best_32.pt` to backend root
6. Copy the ngrok URL → GeoPINN Studio Settings → Colab URL

### GPU Detection (Local)
```python
import torch
print(torch.cuda.is_available())   # True if GPU detected
print(torch.cuda.get_device_name(0))
```

---

## Roadmap

### v4.1
- [ ] 64³ GeoUNet retrain (higher IoU target)
- [ ] CSAMT channel in GeoUNet input
- [ ] IP/SP field data CSV → 3D grid (RBF interpolation)
- [ ] GeoTIFF / Shapefile export (QGIS integration)

### v4.2
- [ ] Docker CPU backend (`docker-compose up`)
- [ ] `pip install geopinn` Python package
- [ ] Benchmark test suite (GitHub Actions)

### v5.0
- [ ] Modal.com GPU cloud integration
- [ ] Physics-Informed loss (PDE residual in training)
- [ ] 2.5D CSAMT forward operator

---

## Citation

```bibtex
@software{geopinn_studio_2026,
  title   = {GeoPINN Studio: Applied Geophysics Desktop Suite with GeoUNet Learned Inversion},
  version = {4.0.0-beta},
  year    = {2026},
  url     = {https://github.com/bilaltlc/geopinn-studio},
  note    = {Beylikova REE-F-Ba-Th exploration analogue · GeoUNet val_iou=0.5703}
}
```

---

## License

MIT License — see [LICENSE](LICENSE)

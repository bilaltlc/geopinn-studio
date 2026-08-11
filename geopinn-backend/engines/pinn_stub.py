"""
pinn_stub.py — GeoUNet 3D U-Net Inference Modülü
GeoPINN Studio v3.2.0

Eğitim kodu analizi:
  - Giriş: gz(32,32) + mag(32,32)  — grid boyutuyla aynı, 21×21 DEĞİL
  - Norm: GroupNorm(min(8,ch), ch) + GELU
  - ObsEnc: Conv2d(2→32→64) + Conv2d(64, 64*g, 1) → view(B,64,g,H,W) → (B,64,32,32,32)
  - val_iou=0.5703  val_mae=0.0221  epoch=204
"""

from __future__ import annotations
import os, warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_OUT = 32
OBS_GRID = 21   # server forward çıkışı — modele vermeden önce 32×32'ye resize edilir
CHECKPOINT_NAMES = [
    "geopinn_unet_best_32.pt",
    "geopinn_unet_best_32.pth",
    "geopinn_unet_32.pt",
    "geopinn_unet.pt",
]

# ── Mimari (eğitim koduyla birebir) ──────────────────────────────────────────

class _CB(nn.Module):
    """Conv3d → GroupNorm(min(8,ch)) → GELU  ×2
    state_dict: .net.0 .net.1 .net.3 .net.4  (GELU index'e girmez)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class _ObsEnc(nn.Module):
    """enc_obs:
      enc: Conv2d(2,32,3,pad=1) + GELU + Conv2d(32,64,3,pad=1) + GELU
      to3: Conv2d(64, 64*g, 1)
      forward: (B,2,g,g) → enc → to3 → view(B,64,g,g,g)
    """
    def __init__(self, g=32):
        super().__init__()
        self.g = g
        self.enc = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
        )
        self.to3 = nn.Conv2d(64, 64 * g, 1)

    def forward(self, gz, mag):
        x = torch.stack([gz, mag], dim=1)           # (B,2,g,g)
        x = self.enc(x)                              # (B,64,g,g)
        x = self.to3(x)                              # (B,64*g,g,g)
        B, _, H, W = x.shape
        return x.view(B, 64, self.g, H, W)          # (B,64,32,32,32)


class GeoUNet(nn.Module):
    def __init__(self, g=32, ch=32):
        super().__init__()
        self.enc_obs = _ObsEnc(g)
        self.e1  = _CB(64,    ch)
        self.e2  = _CB(ch,    ch*2)
        self.e3  = _CB(ch*2,  ch*4)
        self.pool = nn.MaxPool3d(2)
        self.bot = _CB(ch*4,  ch*8)
        self.u3  = nn.ConvTranspose3d(ch*8, ch*4, 2, stride=2)
        self.d3  = _CB(ch*8,  ch*4)
        self.u2  = nn.ConvTranspose3d(ch*4, ch*2, 2, stride=2)
        self.d2  = _CB(ch*4,  ch*2)
        self.u1  = nn.ConvTranspose3d(ch*2, ch,   2, stride=2)
        self.d1  = _CB(ch*2,  ch)
        self.out = nn.Sequential(nn.Conv3d(ch, 1, 1), nn.Sigmoid())

    def forward(self, gz, mag):
        # gz, mag: (B, 32, 32)  — 32×32 giriş bekleniyor
        x  = self.enc_obs(gz, mag)                          # (B,64,32,32,32)
        e1 = self.e1(x)                                     # (B,32,32,32,32)
        e2 = self.e2(self.pool(e1))                         # (B,64,16,16,16)
        e3 = self.e3(self.pool(e2))                         # (B,128,8,8,8)
        b  = self.bot(self.pool(e3))                        # (B,256,4,4,4)
        x  = self.d3(torch.cat([self.u3(b),  e3], dim=1))  # (B,128,8,8,8)
        x  = self.d2(torch.cat([self.u2(x),  e2], dim=1))  # (B,64,16,16,16)
        x  = self.d1(torch.cat([self.u1(x),  e1], dim=1))  # (B,32,32,32,32)
        return self.out(x).squeeze(1)                       # (B,32,32,32)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _find_checkpoint(backend_dir=None):
    dirs = []
    if backend_dir: dirs.append(Path(backend_dir))
    dirs += [Path(__file__).parent.parent, Path.cwd()]
    for d in dirs:
        for name in CHECKPOINT_NAMES:
            p = d / name
            if p.exists(): return p
    return None


_MODEL_CACHE: dict = {}


def _load_model(backend_dir=None, force=False):
    global _MODEL_CACHE
    if _MODEL_CACHE and not force:
        if _MODEL_CACHE.get("available"): return _MODEL_CACHE
        if not backend_dir: return _MODEL_CACHE

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _find_checkpoint(backend_dir)

    if ckpt_path is None:
        _MODEL_CACHE = {
            "model": None, "device": str(device), "checkpoint": None,
            "available": False,
            "error": (f"Checkpoint bulunamadı. Beklenen: {CHECKPOINT_NAMES}. "
                      "Dosyayı geopinn-backend/ klasörüne koyun."),
        }
        return _MODEL_CACHE

    try:
        model = GeoUNet(g=GRID_OUT, ch=32).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd    = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=True)
        model.eval()
        _MODEL_CACHE = {
            "model": model, "device": str(device),
            "checkpoint": str(ckpt_path),
            "checkpoint_size_mb": round(ckpt_path.stat().st_size / 1024**2, 1),
            "available": True, "error": None,
            "meta": {k: v for k, v in state.items()
                     if k != "model_state_dict" and not isinstance(v, torch.Tensor)}
                    if isinstance(state, dict) else {},
        }
    except Exception as e:
        _MODEL_CACHE = {
            "model": None, "device": str(device), "checkpoint": str(ckpt_path),
            "available": False, "error": f"Checkpoint yüklenemedi: {e}",
        }
    return _MODEL_CACHE


# ── Public API ────────────────────────────────────────────────────────────────

def pinn_status(backend_dir=None):
    c = _load_model(backend_dir)
    return {
        "available": c["available"],
        "version": "3.2.0",
        "architecture": "GeoUNet (ObsEnc-2DCNN + 3D U-Net + GroupNorm + GELU)",
        "grid_size": GRID_OUT,
        "input_obs_size": GRID_OUT,   # 32×32 — eğitim boyutu
        "iou_score": 0.5703,
        "val_mae": 0.0221,
        "training_platform": "Kaggle T4",
        "checkpoint": c.get("checkpoint"),
        "checkpoint_size_mb": c.get("checkpoint_size_mb"),
        "device": c["device"],
        "error": c.get("error"),
        "meta": c.get("meta", {}),
    }


def infer(gz_map, mag_map, backend_dir=None, threshold=0.35):
    """
    gz_map  : (21,21) veya (32,32) — her iki boyut kabul edilir, 32×32'ye resize edilir
    mag_map : aynı
    """
    cache = _load_model(backend_dir)
    if not cache["available"]:
        raise RuntimeError(cache["error"])

    model  = cache["model"]
    device = torch.device(cache["device"])

    # Eğitimde kullanılan std normalizasyon sabitleri
    # gen_dataset: G_tr/=gz_std, M_tr/=mag_std  (seed=42, N=500 ile hesaplandı)
    GZ_STD  = 5.97052886e-07
    MAG_STD = 6.72827341e-02

    def _prep(a, std):
        """Std normalize et + 32×32'ye resize (eğitimle aynı pipeline)."""
        a = np.asarray(a, dtype=np.float64).astype(np.float32)
        a = a / std
        t = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        if t.shape[-1] != GRID_OUT or t.shape[-2] != GRID_OUT:
            t = F.interpolate(t, size=(GRID_OUT, GRID_OUT), mode='bilinear', align_corners=False)
        return t.squeeze(0).to(device)                       # (1,32,32)

    gz_t  = _prep(gz_map,  GZ_STD)
    mag_t = _prep(mag_map, MAG_STD)

    with torch.no_grad():
        pred = model(gz_t, mag_t).squeeze(0).cpu().numpy()  # (32,32,32)

    mask       = (pred >= threshold).astype(np.float32)
    voxel_vol  = (480.0 / GRID_OUT) ** 3
    ore_voxels = int(mask.sum())

    return {
        "model_data": pred.tolist(),
        "mask_data":  mask.tolist(),
        "stats": {
            "min": float(pred.min()), "max": float(pred.max()),
            "mean": float(pred.mean()),
            "ore_voxels": ore_voxels,
            "ore_volume_m3": round(ore_voxels * voxel_vol, 0),
            "ore_fraction": round(float(mask.mean()), 4),
            "threshold_used": threshold,
            "grid_size": GRID_OUT,
        },
        "checkpoint": cache.get("checkpoint"),
        "checkpoint_size_mb": cache.get("checkpoint_size_mb"),
        "device": cache["device"],
    }


PINN_AVAILABLE = True
PINN_VERSION   = "3.2.0"
PINN_ROADMAP   = "v4.0.0"
def pinn_forward_stub(*a, **kw):
    raise NotImplementedError("/api/pinn/infer kullanın.")


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    print("[TEST] GeoUNet mimari + checkpoint testi")
    m = GeoUNet(g=32, ch=32)
    # Key kontrolü
    keys = list(m.state_dict().keys())
    print(f"  Toplam key: {len(keys)}")
    print(f"  İlk 6: {keys[:6]}")
    if ckpt:
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = state.get('model_state_dict', state)
        missing, unexpected = m.load_state_dict(sd, strict=True)
        print(f"  Missing   : {missing}")
        print(f"  Unexpected: {unexpected}")
        print(f"  ✓ Checkpoint yüklendi" if not missing and not unexpected else "  ✗ Hata")
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1,32,32), torch.randn(1,32,32))
    print(f"  Çıkış : {tuple(out.shape)}")
    print(f"  Aralık: [{out.min():.4f}, {out.max():.4f}]")

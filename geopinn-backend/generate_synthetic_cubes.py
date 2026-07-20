"""
generate_synthetic_cubes.py — server.py ile UYUMLU (NxNxN küp) sentetik veri üretir.

Kullanım:
    python generate_synthetic_cubes.py --n 20 --grid 64 --out Y_synthetic_v1.npy

Çıktı: (n, grid, grid, grid) şekilli .npy — Veri Seti panelinden yüklendiğinde
Gravite/Manyetik/CSAMT ve Joint Inversion'da doğrudan Y_beylikova.npy gibi çalışır.
"""
import argparse
import numpy as np


def make_body(grid, rng):
    m = np.zeros((grid, grid, grid))
    n_blobs = rng.integers(1, 4)  # 1-3 gövde: tek cisim değil, gerçekçi çoklu-anomali
    for _ in range(n_blobs):
        cx, cy = rng.uniform(0.3, 0.7, 2) * grid
        cz = rng.uniform(0.2, 0.6) * grid          # sığ-orta derinlik
        rx, ry = rng.uniform(0.06, 0.15, 2) * grid
        rz = rng.uniform(0.04, 0.10) * grid
        amp = rng.uniform(0.6, 1.0)
        xx, yy, zz = np.meshgrid(np.arange(grid), np.arange(grid), np.arange(grid), indexing="ij")
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2
        m += amp * np.exp(-d)
    return np.clip(m, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="örnek sayısı")
    p.add_argument("--grid", type=int, default=64, help="küp boyutu (Y_beylikova ile aynı: 64)")
    p.add_argument("--out", type=str, default="Y_synthetic_v1.npy")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    data = np.stack([make_body(args.grid, rng) for _ in range(args.n)], axis=0)
    np.save(args.out, data.astype(np.float64))
    print(f"Yazıldı: {args.out}  şekil={data.shape}  boyut={data.nbytes/1e6:.1f} MB")


if __name__ == "__main__":
    main()

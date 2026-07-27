"""
fvm_core.py — Genel amaçlı Sonlu Hacimler (Finite Volume) Poisson çözücü.

Kullanım alanı: potansiyel-alan (gravite, manyetik skaler potansiyel) fizikleri
için, mevcut analitik prizma motorlarının (gravity_prism, magnetic_prism)
varsaydığı "sonsuz homojen uzay" yerine SINIRLI (bounded) domain'de gerçek
Poisson denklemi çözümü sağlar. Domain sınırında U=0 (Dirichlet) uygulanır —
bu, sonsuzda potansiyelin sönümlenmesinin sonlu-domain yaklaşımıdır.

Not: Bu modül torch KULLANMAZ — saf numpy/scipy.sparse. Sebep: doğrusal
sistem çözümü (spsolve) autograd zinciri gerektirmiyor (mevcut joint
inversion SPSA — gradyansız — kullanıyor, bkz. server.py). İleride analitik
gradyan gerekirse, adjoint-state yöntemiyle ayrı bir backward eklenebilir.

v1 KAPSAMI: sınırlı domain (bounded), düz/varsayılan yüzey. Gerçek DEM
tabanlı topografya (hücre-bazlı hava/yeraltı sınıflandırması) kapsam dışı,
bu çekirdek üzerine ayrı bir adımda eklenecek.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

G_CONST = 6.6743e-11  # m^3 kg^-1 s^-2


def pad_axis(c: np.ndarray, d: float, n_pad: int) -> np.ndarray:
    """Bir eksen dizisini (c), her iki yönde n_pad hücre kadar eşit aralıkla genişletir."""
    left = c[0] - d * np.arange(n_pad, 0, -1)
    right = c[-1] + d * np.arange(1, n_pad + 1)
    return np.concatenate([left, c, right])


def build_padded_grid(x_c: np.ndarray, y_c: np.ndarray, z_c: np.ndarray, pad_cells: int = 8):
    """İç (forward) grid'i sınır etkisini azaltmak için her yönde pad_cells kadar genişletir.

    z ekseni bu projede aşağı-pozitif (z=0 ~ yüzey, +z derinlik). Üst dolgu
    (z<0) 'hava' katmanını, alt dolgu ise domain'in derinlere genişlemesini temsil eder.
    """
    dx = float(x_c[1] - x_c[0])
    dy = float(y_c[1] - y_c[0])
    dz = float(z_c[1] - z_c[0])

    x_p = pad_axis(x_c, dx, pad_cells)
    y_p = pad_axis(y_c, dy, pad_cells)
    z_p = pad_axis(z_c, dz, pad_cells)

    return x_p, y_p, z_p, dx, dy, dz


def embed_model(model_inner: np.ndarray, pad_cells: int) -> np.ndarray:
    """İç modeli (nx,ny,nz), her yönde pad_cells sıfır-dolgu ile büyük domain'e yerleştirir."""
    return np.pad(model_inner, pad_width=pad_cells, mode="constant", constant_values=0.0)


def assemble_poisson_7point(nx: int, ny: int, nz: int, dx: float, dy: float, dz: float) -> sp.csr_matrix:
    """Düzenli grid üzerinde 7-nokta Laplacian'ı seyrek matris olarak kurar (Dirichlet BC henüz yok)."""
    ax, ay, az = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2
    diag_val = -2.0 * (ax + ay + az)

    N = nx * ny * nz
    idx = np.arange(N).reshape(nx, ny, nz)

    rows, cols, vals = [], [], []

    def add_neighbor(shift, axis, coeff):
        sl_self = [slice(None)] * 3
        sl_nbr = [slice(None)] * 3
        if shift > 0:
            sl_self[axis] = slice(0, -1)
            sl_nbr[axis] = slice(1, None)
        else:
            sl_self[axis] = slice(1, None)
            sl_nbr[axis] = slice(0, -1)
        p = idx[tuple(sl_self)].ravel()
        q = idx[tuple(sl_nbr)].ravel()
        rows.extend(p.tolist())
        cols.extend(q.tolist())
        vals.extend([coeff] * len(p))

    # Ana köşegen
    rows.extend(idx.ravel().tolist())
    cols.extend(idx.ravel().tolist())
    vals.extend([diag_val] * N)

    # Komşular (±x, ±y, ±z)
    add_neighbor(+1, 0, ax); add_neighbor(-1, 0, ax)
    add_neighbor(+1, 1, ay); add_neighbor(-1, 1, ay)
    add_neighbor(+1, 2, az); add_neighbor(-1, 2, az)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return A


def apply_dirichlet_boundary(A: sp.csr_matrix, rhs: np.ndarray, nx: int, ny: int, nz: int) -> tuple:
    """Domain sınırındaki hücrelerde U=0 sabitler.

    Verimlilik notu: satır bazlı fancy-indexing (A[b_idx,:]=0) scipy.sparse.lil'de
    dense bir ara matris oluşturup büyük domain'lerde belleği patlatıyor.
    Bunun yerine diagonal-mask ile seyrek-seyrek çarpım kullanılıyor.
    """
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    boundary = np.zeros((nx, ny, nz), dtype=bool)
    boundary[0, :, :] = boundary[-1, :, :] = True
    boundary[:, 0, :] = boundary[:, -1, :] = True
    boundary[:, :, 0] = boundary[:, :, -1] = True
    b_mask = boundary.ravel()
    interior_mask = ~b_mask

    D_int = sp.diags(interior_mask.astype(np.float64))
    D_bnd = sp.diags(b_mask.astype(np.float64))
    A_new = (D_int @ A + D_bnd).tocsr()

    rhs = rhs.copy()
    rhs[b_mask] = 0.0
    return A_new, rhs


def solve_poisson_field(rhs_padded: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Genel ∇²U = rhs çözücü (kaynak terimi dışarıdan verilir). U'yu (nx,ny,nz) döndürür."""
    nx, ny, nz = rhs_padded.shape
    N = nx * ny * nz
    A = assemble_poisson_7point(nx, ny, nz, dx, dy, dz)
    rhs = rhs_padded.ravel().copy()
    A, rhs = apply_dirichlet_boundary(A, rhs, nx, ny, nz)

    if N <= 40_000:
        U = spla.spsolve(A.tocsc(), rhs)
    else:
        diag = A.diagonal()
        diag[diag == 0] = 1.0
        M = spla.LinearOperator(A.shape, matvec=lambda v: v / diag)
        U, info = spla.cg(A, rhs, M=M, rtol=1e-8, maxiter=2000)
        if info != 0:
            raise RuntimeError(f"FVM CG çözücü yakınsamadı (info={info}).")
    return U.reshape(nx, ny, nz)


def solve_poisson_gravity(density_padded: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """∇²U = 4πGρ denklemini çözer, U'yu (nx,ny,nz) şeklinde döndürür (SI birim, m^2/s^2)."""
    rhs = 4.0 * np.pi * G_CONST * density_padded
    return solve_poisson_field(rhs, dx, dy, dz)


def gradient_at_surface(U: np.ndarray, pad_cells: int, dx: float, dy: float, dz: float):
    """Yüzeyde (z=0, dolgu/iç grid arayüzü) tam gradyanı (dU/dx, dU/dy, dU/dz) döndürür."""
    U_at_surface = 0.5 * (U[:, :, pad_cells] + U[:, :, pad_cells - 1])
    dUdz = (U[:, :, pad_cells] - U[:, :, pad_cells - 1]) / dz
    dUdx = np.gradient(U_at_surface, dx, axis=0)
    dUdy = np.gradient(U_at_surface, dy, axis=1)
    return dUdx, dUdy, dUdz


def vertical_gradient_at_surface(U: np.ndarray, pad_cells: int, dz: float) -> np.ndarray:
    """U alanının gerçek yüzeyde (z=0, dolgu/iç grid arayüzü) dikey türevini döndürür.

    z_p[pad_cells-1] ve z_p[pad_cells] tam olarak z=0'ın ±dz/2 komşularıdır;
    aralarındaki fark, MERKEZİ FARK anlamında tam z=0'da ikinci-mertebe
    doğru bir türev tahmini verir (vertical_gradient_at_index'teki gibi bir
    hücre merkezinde DEĞİL, iki hücre arasındaki arayüzde değerlendirilir).
    gz sözleşimi: aşağı-pozitif (gravity_prism.py ile TUTARLI).
    """
    dUdz = (U[:, :, pad_cells] - U[:, :, pad_cells - 1]) / dz
    return -dUdz


def vertical_gradient_at_index(U: np.ndarray, z_index: int, dz: float) -> np.ndarray:
    """U alanının z_index katmanındaki dikey türevini (merkezi fark) (nx,ny) olarak döndürür.
    gz sözleşimi: aşağı-pozitif (mevcut gravity_prism.py ile TUTARLI)."""
    z_index = int(np.clip(z_index, 1, U.shape[2] - 2))
    dUdz = (U[:, :, z_index + 1] - U[:, :, z_index - 1]) / (2.0 * dz)
    return -dUdz  # yukarı yönlü potansiyel gradyanının negatifi = aşağı-pozitif çekim ivmesi

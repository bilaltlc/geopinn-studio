"""
magnetic_fvm.py — Sınırlı-domain FVM tabanlı manyetik (TMI) ileri yönlü modelleme.

magnetic_prism.PrismMagneticForward ile AYNI .calculate() imzasını taşır.

Fizik: manyetik skaler potansiyel phi_m için ∇²phi_m = ∇·M, M = chi(x)*H0.
H = -∇phi_m, B = mu0*H (yüzeyde chi≈0 varsayımıyla M=0), ΔT = f̂·B (nT).
"""

import numpy as np
from . import fvm_core

MU_0 = 4 * np.pi * 1e-7


class PrismMagneticForwardFVM:
    def __init__(self, inc_deg=60.0, dec_deg=5.0, b0_nt=47000.0, pad_cells: int = 16):
        self.pad_cells = pad_cells
        inc = np.radians(inc_deg)
        dec = np.radians(dec_deg)
        self.L = np.cos(inc) * np.cos(dec)
        self.M = np.cos(inc) * np.sin(dec)
        self.N = np.sin(inc)
        self.H0_mag = (b0_nt * 1e-9) / MU_0  # A/m

    def calculate(self, chi_matrix, x_coords, y_coords, z_coords,
                   obs_x, obs_y, obs_z=0.0, **kwargs):
        chi_matrix = np.asarray(chi_matrix, dtype=np.float64)
        x_coords = np.asarray(x_coords, dtype=np.float64)
        y_coords = np.asarray(y_coords, dtype=np.float64)
        z_coords = np.asarray(z_coords, dtype=np.float64)

        x_p, y_p, z_p, dx, dy, dz = fvm_core.build_padded_grid(
            x_coords, y_coords, z_coords, pad_cells=self.pad_cells
        )
        chi_p = fvm_core.embed_model(chi_matrix, self.pad_cells)

        gchi_x = np.gradient(chi_p, dx, axis=0)
        gchi_y = np.gradient(chi_p, dy, axis=1)
        gchi_z = np.gradient(chi_p, dz, axis=2)

        rhs = self.H0_mag * (self.L * gchi_x + self.M * gchi_y + self.N * gchi_z)

        phi = fvm_core.solve_poisson_field(rhs, dx, dy, dz)

        dUdx, dUdy, dUdz = fvm_core.gradient_at_surface(phi, self.pad_cells, dx, dy, dz)
        Bx = -MU_0 * dUdx * 1e9
        By = -MU_0 * dUdy * 1e9
        Bz = -MU_0 * dUdz * 1e9
        dT_full = self.L * Bx + self.M * By + self.N * Bz

        interp = _RegularGridInterp2D(x_p, y_p, dT_full)
        dt_obs = interp(obs_x, obs_y)
        return dt_obs.reshape(obs_x.shape)


class _RegularGridInterp2D:
    def __init__(self, x, y, values):
        from scipy.interpolate import RegularGridInterpolator
        self._interp = RegularGridInterpolator((x, y), values, bounds_error=False, fill_value=0.0)

    def __call__(self, obs_x, obs_y):
        pts = np.column_stack([obs_x.ravel(), obs_y.ravel()])
        return self._interp(pts)

"""
gravity_fvm.py — Sınırlı-domain (bounded) FVM tabanlı gravite ileri yönlü modelleme.

gravity_prism.PrismGravityForward ile AYNI .calculate() imzasını taşır —
server.py'de doğrudan yerine (ya da opsiyonel bir "engine_mode" anahtarıyla
yanına) takılabilir.

Fark: gravity_prism sonsuz-homojen-uzay Green fonksiyonu (Nagy kernel) ile
kapalı-form toplama yapar; bu motor gerçek Poisson denklemini (∇²U=4πGρ)
sınırlı domain'de, Dirichlet (U=0) sınır koşuluyla nümerik çözer.
"""

import numpy as np
from . import fvm_core


class PrismGravityForwardFVM:
    def __init__(self, pad_cells: int = 8):
        self.SI_TO_MGAL = 1e5
        self.pad_cells = pad_cells

    def calculate(self, density_contrast_kg_m3, x_coords, y_coords, z_coords,
                   obs_x, obs_y, obs_z=0.0, **kwargs):
        density_contrast_kg_m3 = np.asarray(density_contrast_kg_m3, dtype=np.float64)
        x_coords = np.asarray(x_coords, dtype=np.float64)
        y_coords = np.asarray(y_coords, dtype=np.float64)
        z_coords = np.asarray(z_coords, dtype=np.float64)

        x_p, y_p, z_p, dx, dy, dz = fvm_core.build_padded_grid(
            x_coords, y_coords, z_coords, pad_cells=self.pad_cells
        )
        rho_p = fvm_core.embed_model(density_contrast_kg_m3, self.pad_cells)

        U = fvm_core.solve_poisson_gravity(rho_p, dx, dy, dz)

        # Gerçek yüzey (z=0), dolgu/iç grid arayüzünde tam olarak değerlendirilir.
        gz_full = fvm_core.vertical_gradient_at_surface(U, self.pad_cells, dz) * self.SI_TO_MGAL

        interp = _RegularGridInterp2D(x_p, y_p, gz_full)
        gz_obs = interp(obs_x, obs_y)
        return gz_obs.reshape(obs_x.shape)


class _RegularGridInterp2D:
    """scipy bağımlılığını tek noktada tutmak için ince bir sarmalayıcı."""
    def __init__(self, x, y, values):
        from scipy.interpolate import RegularGridInterpolator
        self._interp = RegularGridInterpolator((x, y), values, bounds_error=False, fill_value=0.0)

    def __call__(self, obs_x, obs_y):
        pts = np.column_stack([obs_x.ravel(), obs_y.ravel()])
        return self._interp(pts)

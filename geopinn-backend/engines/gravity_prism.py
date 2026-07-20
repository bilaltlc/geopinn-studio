import torch
import time
import numpy as np

class PrismGravityForward:
    """
    Dikdörtgen prizmalar için analitik gravite ileri yönlü modelleme motoru.
    (PyTorch ile %100 GPU Hızlandırmalı, OOM Korumalı ve Autograd Kontrollü)
    """
    def __init__(self):
        self.G = 6.6743e-11
        self.SI_TO_MGAL = 1e5
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BİLGİ] Hesaplama donanımı: {self.device}".upper())

    def _nagy_kernel(self, dx, dy, dz):
        r = torch.sqrt(dx**2 + dy**2 + dz**2)
        eps = 1e-12 

        term1 = dx * torch.log(dy + r + eps)
        term2 = dy * torch.log(dx + r + eps)
        term3 = -dz * torch.atan2(dx * dy, dz * r + eps)

        return term1 + term2 + term3

    def calculate(self, density_contrast_kg_m3, x_coords, y_coords, z_coords, obs_x, obs_y, obs_z=0.0, batch_size=10, track_gradients=False):
        if isinstance(density_contrast_kg_m3, torch.Tensor):
            rho_tensor = density_contrast_kg_m3.to(self.device, dtype=torch.float64)
        else:
            rho_tensor = torch.tensor(density_contrast_kg_m3, dtype=torch.float64, device=self.device)
            
        x_tensor = torch.tensor(x_coords, dtype=torch.float64, device=self.device)
        y_tensor = torch.tensor(y_coords, dtype=torch.float64, device=self.device)
        z_tensor = torch.tensor(z_coords, dtype=torch.float64, device=self.device)
        
        obs_x_tensor = torch.tensor(obs_x.flatten(), dtype=torch.float64, device=self.device)
        obs_y_tensor = torch.tensor(obs_y.flatten(), dtype=torch.float64, device=self.device)
        obs_z_tensor = torch.tensor(obs_z, dtype=torch.float64, device=self.device)

        active_mask = torch.abs(rho_tensor) > 1e-3
        rho_active = rho_tensor[active_mask]
        
        if len(rho_active) == 0:
            return torch.zeros_like(torch.tensor(obs_x), dtype=torch.float64, device=self.device)

        X, Y, Z = torch.meshgrid(x_tensor, y_tensor, z_tensor, indexing='ij')
        x_act = X[active_mask]
        y_act = Y[active_mask]
        z_act = Z[active_mask]

        dx_cell = x_coords[1] - x_coords[0]
        dy_cell = y_coords[1] - y_coords[0]
        dz_cell = z_coords[1] - z_coords[0]

        x_min, x_max = x_act - dx_cell/2, x_act + dx_cell/2
        y_min, y_max = y_act - dy_cell/2, y_act + dy_cell/2
        z_min, z_max = z_act - dz_cell/2, z_act + dz_cell/2

        N_obs = len(obs_x_tensor)
        gz_chunks = []
        
        with torch.set_grad_enabled(track_gradients):
            for start_idx in range(0, N_obs, batch_size):
                end_idx = min(start_idx + batch_size, N_obs)
                
                ox = obs_x_tensor[start_idx:end_idx, None] 
                oy = obs_y_tensor[start_idx:end_idx, None]
                oz = obs_z_tensor
                
                dx1 = x_min[None, :] - ox
                dx2 = x_max[None, :] - ox
                dy1 = y_min[None, :] - oy
                dy2 = y_max[None, :] - oy
                dz1 = z_min[None, :] - oz
                dz2 = z_max[None, :] - oz

                # İŞARET DÜZELTİLDİ: Aşağı yönlü pozitif çekim için -1.0 eklendi.
                sum_gz = -1.0 * (
                      self._nagy_kernel(dx2, dy2, dz2)
                    - self._nagy_kernel(dx1, dy2, dz2)
                    - self._nagy_kernel(dx2, dy1, dz2)
                    + self._nagy_kernel(dx1, dy1, dz2)
                    - self._nagy_kernel(dx2, dy2, dz1)
                    + self._nagy_kernel(dx1, dy2, dz1)
                    + self._nagy_kernel(dx2, dy1, dz1)
                    - self._nagy_kernel(dx1, dy1, dz1)
                )

                gz_chunks.append(torch.sum(self.G * rho_active * sum_gz, dim=1))

                del sum_gz, dx1, dx2, dy1, dy2, dz1, dz2, ox, oy
                if not track_gradients:
                    torch.cuda.empty_cache()

        gz_flat = torch.cat(gz_chunks, dim=0)
        gz_flat = gz_flat * self.SI_TO_MGAL
        return gz_flat.reshape(obs_x.shape)


def run_gravity_forward(density=1000.0, depth=30.0):
    """
    API tarafindan cagirilan, 3D uzayi anlik olusturup motoru calistiran kopru.
    """
    import time
    
    # 1. Uzay ve Yoğunluk Modeli (100x100x100m'lik bir yeralti hacmi, 10'ar metre grid)
    coords_x = np.linspace(0, 100, 11)
    coords_y = np.linspace(0, 100, 11)
    coords_z = np.linspace(0, 100, 11)
    
    mock_rho = np.zeros((11, 11, 11))
    
    # Derinlik indeksini hesapla (0-100m arasi, Z ekseni asagi dogru)
    z_idx = int(np.clip(depth / 10.0, 0, 10))
    
    # Uzayin ortasina (X=4-6, Y=4-6) istenen derinlikte (z_idx) bir prizmatik cevher yerlestir
    z_end = min(z_idx + 3, 11) 
    mock_rho[4:7, 4:7, z_idx:z_end] = density 
    
    # 2. Yüzeydeki Gözlem Ağı (21x21 grid istasyonu)
    obs_coords = np.linspace(0, 100, 21)
    ox, oy = np.meshgrid(obs_coords, obs_coords)
    
    # Senin Pytorch Sinifini Baslat
    engine = PrismGravityForward()
    start_time = time.time()
    
    # Motoru calistir (Gradients=False cunku sadece ileri yonlu simülasyon yapiyoruz)
    gz = engine.calculate(mock_rho, coords_x, coords_y, coords_z, ox, oy, batch_size=50, track_gradients=False)
    
    calc_time = time.time() - start_time
    max_anom = float(gz.max())
    
    return {
        "status": "success",
        "message": f"PyTorch Shape: {list(gz.shape)}",
        "max_anom": round(max_anom, 4),
        "calc_time": round(calc_time, 4),
        "misfit": 0.0
    }
import torch
import numpy as np
import time

class PrismMagneticForward:
    """
    Dikdörtgen prizmalar için Bhattacharyya (1964) analitik toplam alan manyetik
    (Total Magnetic Intensity - TMI) ileri yönlü modelleme motoru.
    (PyTorch ile %100 GPU Hızlandırmalı, OOM Korumalı ve Autograd Kontrollü)
    """
    def __init__(self, inc_deg=60.0, dec_deg=5.0, b0_nt=47000.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BİLGİ] Manyetik Hesaplama Donanımı: {self.device}".upper())

        self.inc = np.radians(inc_deg)
        self.dec = np.radians(dec_deg)
        self.b0 = b0_nt
        self.CM = 1.0
        
        # Yön kosinüslerini hesaplayıp doğrudan GPU tensörü olarak saklıyoruz
        self.L = torch.tensor(np.cos(self.inc) * np.cos(self.dec), dtype=torch.float64, device=self.device)
        self.M = torch.tensor(np.cos(self.inc) * np.sin(self.dec), dtype=torch.float64, device=self.device)
        self.N = torch.tensor(np.sin(self.inc), dtype=torch.float64, device=self.device)

    def _bhattacharyya_kernel(self, x, y, z):
        """Bhattacharyya analitik prizma integrali çekirdek fonksiyonu (GPU Tensör Uyarlaması)."""
        r = torch.sqrt(x**2 + y**2 + z**2)
        eps = 1e-12
        
        term1 = 0.5 * (self.N**2 - self.L**2) * torch.log(r - y + eps)
        term2 = 0.5 * (self.N**2 - self.M**2) * torch.log(r - x + eps)
        term3 = -self.L * self.M * torch.log(r + z + eps)
        term4 = -self.L * self.N * torch.atan2(x * y, (x**2 + r * z + eps))
        term5 = -self.M * self.N * torch.atan2(x * y, (y**2 + r * z + eps))
        
        return term1 + term2 + term3 + term4 + term5

    def calculate(self, chi_matrix, x_coords, y_coords, z_coords, obs_x, obs_y, obs_z=0.0, batch_size=10, track_gradients=False):
        """
        Filtrelenmiş aktif prizmalar üzerinden yüzey TMI anomalisini (nT) hesaplar.
        
        track_gradients: Sadece veri üretimi (forward) yapılıyorsa False kalarak RAM korur.
                         PINN ile eğitim esnasında ters çözüm yapılıyorsa True yapılmalıdır.
        """
        # Gelen chi GIRDISI: ayni gradyan-koruma mantigi gravity_prism.py'deki
        # gibi -- tensor ise dogrudan kullan (gecmis korunur), degilse yarat.
        if isinstance(chi_matrix, torch.Tensor):
            chi_tensor = chi_matrix.to(self.device, dtype=torch.float64)
        else:
            chi_tensor = torch.tensor(chi_matrix, dtype=torch.float64, device=self.device)
        x_tensor = torch.tensor(x_coords, dtype=torch.float64, device=self.device)
        y_tensor = torch.tensor(y_coords, dtype=torch.float64, device=self.device)
        z_tensor = torch.tensor(z_coords, dtype=torch.float64, device=self.device)
        
        obs_x_tensor = torch.tensor(obs_x.flatten(), dtype=torch.float64, device=self.device)
        obs_y_tensor = torch.tensor(obs_y.flatten(), dtype=torch.float64, device=self.device)
        obs_z_tensor = torch.tensor(obs_z, dtype=torch.float64, device=self.device)

        # Sıfır kontrast maskelemesi (Zayıf sinyalleri korumak için eşik 1e-5 SI)
        active_mask = torch.abs(chi_tensor) > 1e-5
        chi_active = chi_tensor[active_mask]
        
        if len(chi_active) == 0:
            return torch.zeros_like(torch.tensor(obs_x), dtype=torch.float64, device=self.device)

        # 3D Grid oluştur
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
        # ayni guvenlik notu: in-place slice atama yerine liste+cat (gravity_prism.py ile tutarli)
        dt_chunks = []
        
        # OOM Hatasını önlemek için türev takibini şarta bağla ve Mini-Batch yap
        with torch.set_grad_enabled(track_gradients):
            for start_idx in range(0, N_obs, batch_size):
                end_idx = min(start_idx + batch_size, N_obs)
                
                ox = obs_x_tensor[start_idx:end_idx, None] 
                oy = obs_y_tensor[start_idx:end_idx, None]
                oz = obs_z_tensor
                
                # Tüm noktalara göre mesafelerin genişletilmiş tensör matrisleri
                dx1 = x_min[None, :] - ox
                dx2 = x_max[None, :] - ox
                dy1 = y_min[None, :] - oy
                dy2 = y_max[None, :] - oy
                dz1 = z_min[None, :] - oz
                dz2 = z_max[None, :] - oz

                # Bhattacharyya (1964) formülünün 8 köşeli süperpozisyonu
                sum_dt = (
                      self._bhattacharyya_kernel(dx2, dy2, dz2) - self._bhattacharyya_kernel(dx1, dy2, dz2)
                    - self._bhattacharyya_kernel(dx2, dy1, dz2) + self._bhattacharyya_kernel(dx1, dy1, dz2)
                    - self._bhattacharyya_kernel(dx2, dy2, dz1) + self._bhattacharyya_kernel(dx1, dy2, dz1)
                    + self._bhattacharyya_kernel(dx2, dy1, dz1) - self._bhattacharyya_kernel(dx1, dy1, dz1)
                )

                # nT biriminde Toplam Manyetik Alan (TMI) hesabı
                dt_chunks.append(torch.sum(self.CM * chi_active * self.b0 * sum_dt, dim=1))

                # Manuel GPU Bellek (VRAM) Temizliği
                del sum_dt, dx1, dx2, dy1, dy2, dz1, dz2, ox, oy
                if not track_gradients:
                    torch.cuda.empty_cache()

        dt_flat = torch.cat(dt_chunks, dim=0)
        # Sonucu tensor olarak dondur (gradyan zinciri korunur)
        return dt_flat.reshape(obs_x.shape)


if __name__ == "__main__":
    print("[BAŞLADI] PyTorch Manyetik (TMI) ileri yönlü modelleme çalıştırılıyor...", flush=True)
    
    # Basit test grid'i (Gravite testindeki yapıyla aynı)
    coords = np.linspace(0, 100, 10)
    mock_chi = np.zeros((10, 10, 10))
    # Ortaya 3.0e-4 SI (Petrofizik modülündeki manyetik anomali) değerinde blok koyuyoruz
    mock_chi[4:6, 4:6, 4:6] = 3.0e-4 
    
    obs_coords = np.linspace(0, 100, 21)
    ox, oy = np.meshgrid(obs_coords, obs_coords)
    
    engine = PrismMagneticForward(inc_deg=60.0, dec_deg=5.0, b0_nt=47000.0)
    
    start_time = time.time()
    # Belleği yormamak için batch_size 10 olarak test ediliyor
    dt_anom = engine.calculate(mock_chi, coords, coords, coords, ox, oy, batch_size=10)
    end_time = time.time()
    
    print(f"Başarılı! Boyut: {dt_anom.shape}")
    print(f"Hesaplama Süresi: {(end_time - start_time):.4f} saniye")
    print(f"Maksimum Manyetik Anomali: {dt_anom.max():.4f} nT")
    print(f"Minimum Manyetik Anomali: {dt_anom.min():.4f} nT")
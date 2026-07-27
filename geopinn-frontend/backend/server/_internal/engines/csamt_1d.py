import torch
import numpy as np
import time

class CSAMT1DForward:
    """
    1D Magnetotellürik (MT) / Uzak-Alan CSAMT İleri Yönlü Modelleme Motoru.
    (PyTorch ile %100 GPU Hızlandırmalı, Karmaşık Sayı (Complex) Tensör Altyapısı)
    
    Referans: Ward and Hohmann (1988), Electromagnetic theory for geophysical applications.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BİLGİ] CSAMT/MT Hesaplama Donanımı: {self.device}".upper())
        
        # Fiziksel Sabitler
        self.mu_0 = 4 * np.pi * 1e-7  # Boşluğun manyetik geçirgenliği (H/m)

    def calculate(self, frequencies, thicknesses, resistivities, batch_size=2000, track_gradients=False):
        """
        Gelen frekanslar ve 1D yeraltı modelleri için Görünür Özdirenç ve Faz hesaplar.
        
        frequencies: (N_freq,) boyutlu frekans dizisi (Hz)
        thicknesses: (N_layers - 1,) boyutlu katman kalınlıkları dizisi (m)
        resistivities: (N_stations, N_layers) boyutlu istasyonların 1D özdirenç profilleri (Ohm.m)
                       Son katman sonsuz yarı-uzay (basement) kabul edilir.
        batch_size: RAM (VRAM) koruması için aynı anda işlenecek istasyon sayısı.
        """
        # 1. Girdileri GPU Tensörlerine Çevir
        freq_tensor = torch.tensor(frequencies, dtype=torch.float64, device=self.device)
        thick_tensor = torch.tensor(thicknesses, dtype=torch.float64, device=self.device)
        
        # resistivities Numpy array olarak gelirse Tensöre çevir, zaten tensörse cihazı ayarla
        if not isinstance(resistivities, torch.Tensor):
            res_tensor = torch.tensor(resistivities, dtype=torch.float64, device=self.device)
        else:
            res_tensor = resistivities.to(self.device, dtype=torch.float64)

        N_stations, N_layers = res_tensor.shape
        N_freq = len(freq_tensor)
        
        # Açısal frekans: omega = 2 * pi * f
        omega = 2 * torch.pi * freq_tensor  # (N_freq,)
        omega_mu = omega * self.mu_0        # (N_freq,)
        
        # PyTorch'ta karmaşık (complex) sayı oluşturma: i * omega * mu_0
        # Şekil: (1, N_freq) - Broadcasting için hazır
        i_omega_mu = torch.complex(torch.zeros_like(omega_mu), omega_mu).unsqueeze(0)

        # Sonuçları tutacağımız ana matrisler
        # Not: app_res_all/phase_all artik on-tahsis edilmiyor; batch sonuclari
        # listede toplanip donguden sonra torch.cat ile birlestiriliyor.

        with torch.set_grad_enabled(track_gradients):
            # Belleği (VRAM) korumak için mini-batch döngüsü
            # GUVENLIK NOTU: in-place slice atama (app_res_all[start:end,:]=...)
            # yerine liste+cat kullaniliyor (gravity_prism.py/magnetic_prism.py
            # ile tutarli, autograd icin daima guvenli standart yontem).
            app_res_chunks = []
            phase_chunks = []
            for start_idx in range(0, N_stations, batch_size):
                end_idx = min(start_idx + batch_size, N_stations)
                
                # Bu batch'teki istasyonların özdirenç profillerini al: (batch_size, N_layers)
                res_batch = res_tensor[start_idx:end_idx, :]
                
                # --- WAİT'İN ÖZYİNELEME (RECURSION) ALGORİTMASI ---
                
                # Adım 1: En alt katmanın (sonsuz yarı-uzay) empedansını hesapla
                # Z_n = sqrt(i * omega * mu_0 * rho_n)
                rho_n = res_batch[:, -1].unsqueeze(1) # (batch, 1)
                
                # Z karmaşık bir tensördür, boyutu: (batch_size, N_freq)
                Z = torch.sqrt(i_omega_mu * rho_n)
                
                # Adım 2: Aşağıdan yukarıya (yüzeye doğru) katmanları entegre et
                for j in range(N_layers - 2, -1, -1):
                    rho_j = res_batch[:, j].unsqueeze(1)  # (batch, 1)
                    h_j = thick_tensor[j]                 # skaler/tekil tensör
                    
                    # Katmanın intrinsik (öz) empedansı ve yayılım sabiti
                    W_j = torch.sqrt(i_omega_mu * rho_j)
                    gamma_j = torch.sqrt(i_omega_mu / rho_j)
                    
                    # PyTorch karmaşık tensörlerde tanh fonksiyonunu destekler
                    tanh_val = torch.tanh(gamma_j * h_j)
                    
                    # Empedansın yüzeye taşınması
                    numerator = Z + W_j * tanh_val
                    denominator = W_j + Z * tanh_val
                    Z = W_j * (numerator / denominator)
                
                # Adım 3: Yüzey empedansından Görünür Özdirenç ve Faz çıkarımı
                app_res = (1.0 / omega_mu) * torch.abs(Z)**2
                phase = torch.angle(Z) * (180.0 / torch.pi)
                
                # Sonuçları listeye ekle
                app_res_chunks.append(app_res)
                phase_chunks.append(phase)
                
                # Autograd kapalıysa manuel bellek temizliği yap
                if not track_gradients:
                    del res_batch, rho_n, Z, W_j, gamma_j, tanh_val, numerator, denominator, app_res, phase
                    torch.cuda.empty_cache()

        app_res_all = torch.cat(app_res_chunks, dim=0)
        phase_all = torch.cat(phase_chunks, dim=0)

        # Eger türev takibi açıksa tensor döndür (zinciri koparmamak için), değilse numpy döndür
        if track_gradients:
            return app_res_all, phase_all
        else:
            return app_res_all.cpu().numpy(), phase_all.cpu().numpy()


if __name__ == "__main__":
    print("[BAŞLADI] PyTorch 1D CSAMT İleri Yönlü Modelleme çalıştırılıyor...", flush=True)
    
    freqs = np.logspace(4, -1, 20) 
    thick = np.array([100.0, 500.0])
    N_stations = 10000 
    
    res_matrix = np.zeros((N_stations, 3))
    res_matrix[:, 0] = 50.0
    res_matrix[:, 1] = 500.0
    res_matrix[:, 2] = 2000.0
    
    engine = CSAMT1DForward()
    
    start_time = time.time()
    # GPU VRAM'i yormamak için batch_size 2000
    rho_a, phi = engine.calculate(freqs, thick, res_matrix, batch_size=2000)
    end_time = time.time()
    
    print(f"Başarılı! Hesaplanılan İstasyon: {N_stations}, Frekans Sayısı: {len(freqs)}")
    print(f"Çıktı Boyutları -> Görünür Özdirenç: {rho_a.shape}, Faz: {phi.shape}")
    print(f"GPU Hesaplama Süresi: {(end_time - start_time):.4f} saniye")
    print(f"İstasyon 1, Yüksek Frekans (10kHz) G.Özdirenç: {rho_a[0, 0]:.2f} Ohm.m")
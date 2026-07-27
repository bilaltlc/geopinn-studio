"""
forward/petrophysics.py

Normalize edilmis [0,1] geometri alanini (geometry/gaussian_bodies.py ciktisi)
GERCEK fiziksel birimlere cevirir: yogunluk (g/cm3), manyetik duyarlilik (SI),
ozdirenc (Ohm.m). (PyTorch ile %100 GPU Uyumlu)

TEORI - Neden ayri bir katman?
    rho(x) = rho_host + Delta_rho * f_geometri(x)

    f_geometri(x)  : [0,1] araliginda, SADECE SEKIL bilgisi tasir.
    rho_host       : host kayanin zemin degeri
    Delta_rho      : cevher-host arasindaki KONTRAST
"""

from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class PetrophysicalContrast:
    """
    Tek bir fiziksel ozellik (yogunluk VEYA duyarlilik VEYA ozdirenc) icin
    host-zemin degeri ve cevher-kontrasti. PyTorch tensörleriyle donanım bağımsız çalışır.
    """
    host_value: float
    contrast: float           # additive mod: host + contrast*f ; multiplicative mod: host * contrast**f
    mode: str = "additive"    # "additive" (yogunluk/duyarlilik) | "log_multiplicative" (ozdirenc)

    def apply(self, normalized_field: torch.Tensor) -> torch.Tensor:
        """
        Geometri tensörünü (GPU veya CPU'da olabilir) fiziksel alana çevirir.
        İşlem, gelen tensörün bulunduğu cihazda (device) gerçekleşir.
        """
        f = normalized_field
        if self.mode == "additive":
            return self.host_value + self.contrast * f
        elif self.mode == "log_multiplicative":
            # rho(x) = rho_host * (rho_ore/rho_host) ** f(x)
            ratio = self.contrast  
            # PyTorch'ta (float ** tensor) işlemi tensörün cihazını ve gradyanını korur
            return self.host_value * (ratio ** f)
        else:
            raise ValueError(f"Bilinmeyen mode: {self.mode}")


@dataclass
class BeylikovaPetrophysics:
    """
    Beylikova-tipi REE-F-Ba-Th yataginin PoC petrofizik parametre seti.
    """
    # --- Yogunluk (g/cm3) ---
    density_host: float = 2.70
    density_contrast: float = 2.00   

    # --- Manyetik duyarlilik (SI birim, boyutsuz) ---
    susceptibility_host: float = 1.0e-4    
    susceptibility_contrast: float = 3.0e-4  

    # --- Ozdirenc (Ohm.m) ---
    resistivity_host: float = 500.0       
    resistivity_ore_ratio: float = 0.10   

    def density_field(self, normalized_geometry: torch.Tensor) -> torch.Tensor:
        contrast = PetrophysicalContrast(
            host_value=self.density_host,
            contrast=self.density_contrast,
            mode="additive",
        )
        return contrast.apply(normalized_geometry)

    def susceptibility_field(self, normalized_geometry: torch.Tensor) -> torch.Tensor:
        contrast = PetrophysicalContrast(
            host_value=self.susceptibility_host,
            contrast=self.susceptibility_contrast,
            mode="additive",
        )
        return contrast.apply(normalized_geometry)

    def resistivity_field(self, normalized_geometry: torch.Tensor) -> torch.Tensor:
        contrast = PetrophysicalContrast(
            host_value=self.resistivity_host,
            contrast=self.resistivity_ore_ratio,
            mode="log_multiplicative",
        )
        return contrast.apply(normalized_geometry)

    def all_fields(self, normalized_geometry: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "density": self.density_field(normalized_geometry),
            "susceptibility": self.susceptibility_field(normalized_geometry),
            "resistivity": self.resistivity_field(normalized_geometry),
        }

if __name__ == "__main__":
    import time
    print("[BAŞLADI] PyTorch Petrofizik modülü bağımsız testi tetiklendi.", flush=True)
    
    # GPU varsa CUDA kullan, yoksa CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}".upper())
    
    # Test için 80x80x80 boyutunda [0, 1] arası rastgele bir geometri matrisi (GPU üzerinde)
    mock_field = torch.rand((80, 80, 80), dtype=torch.float64, device=device)
    
    start_time = time.time()
    
    petro = BeylikovaPetrophysics()
    fields = petro.all_fields(mock_field)
    
    rho = fields["density"]
    chi = fields["susceptibility"]
    res = fields["resistivity"]
    
    end_time = time.time()
    
    print(f"Başarılı! (Süre: {end_time - start_time:.4f} saniye)")
    print(f"Yoğunluk Sınırları: {rho.min().item():.2f} - {rho.max().item():.2f} g/cm3")
    print(f"Manyetik Duyarlılık Sınırları: {chi.min().item():.5f} - {chi.max().item():.5f} SI")
    print(f"Özdirenç Sınırları: {res.min().item():.2f} - {res.max().item():.2f} Ohm.m")
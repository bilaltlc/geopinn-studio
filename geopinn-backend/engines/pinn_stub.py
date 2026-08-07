"""
pinn_stub.py — Physics-Informed Neural Network (PINN) Modülü
GeoPINN Studio v4.0 Hazırlık

Bu modül v3.x'te PLACEHOLDER olarak bulunur.
v4.0'da tam PINN entegrasyonu planlanmaktadır.

Planlanan mimari:
    forward_physics(model) → observations  [mevcut: analitik/FVM]
    pinn_loss(model, obs)  → physics_loss + data_loss
    pinn_inversion(obs)    → model  [v4.0 hedefi]

Referanslar:
    Raissi et al. (2019) — Physics-informed neural networks
    Sun et al. (2020) — Surrogate modeling for geophysics with PINNs
    Waheed et al. (2021) — PINNeik: Eikonal solution using PINNs
"""

PINN_AVAILABLE = False
PINN_VERSION   = "stub-3.x"
PINN_ROADMAP   = "v4.0.0"


def pinn_status() -> dict:
    return {
        "available":   PINN_AVAILABLE,
        "version":     PINN_VERSION,
        "roadmap":     PINN_ROADMAP,
        "description": "PINN entegrasyonu v4.0'da gelecek",
        "planned_features": [
            "Laplace/Poisson PDE kayıp fonksiyonu",
            "PyTorch autograd fizik katmanı",
            "Çok-yöntemli PINN joint inversion",
            "Transfer learning: sentetik → gerçek saha",
            "Uncertainty quantification via Bayesian PINN",
        ],
        "current_alternative": (
            "Şu an: Adam optimizer + autograd gradient tabanlı joint inversion "
            "(server.py /api/joint-inversion endpoint)"
        ),
    }


def pinn_forward_stub(model_data, method="gravity"):
    """
    PINN ileri model STUB — v4.0'da gerçek implementasyon gelecek.
    Şu an mevcut analitik motorlara yönlendirir.
    """
    raise NotImplementedError(
        f"PINN forward model henüz implement edilmedi (v{PINN_ROADMAP} hedefi). "
        f"Şu an /api/run-physics-engine endpoint'ini kullanın."
    )


if __name__ == "__main__":
    import json
    print(json.dumps(pinn_status(), indent=2, ensure_ascii=False))

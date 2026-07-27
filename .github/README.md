# GeoPINN Studio 3.0

**Uygulamalı Jeofizik Masaüstü Uygulaması** — Gravite, Manyetik ve CSAMT forward modelleme + joint ters çözüm.

## Kurulum (kullanıcı)

[Releases](../../releases) sayfasından son versiyonu indir:
- Windows: `GeoPINN.Studio.Setup.x.x.x.exe`
- macOS: `GeoPINN.Studio-x.x.x.dmg`
- Linux: `GeoPINN.Studio-x.x.x.AppImage`

## Geliştirme

```bash
# Gereksinimler: Node.js 20+, Python 3.11+

# Frontend bağımlılıkları
npm install

# Backend bağımlılıkları
pip install fastapi uvicorn numpy scipy torch simpeg discretize harmonica python-multipart

# Geliştirme modu
npm run dev

# Backend (ayrı terminalde)
cd ../geopinn-backend
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

## Colab GPU Modu

`geopinn_colab_backend.ipynb` dosyasını Colab'a yükle, çalıştır, ngrok URL'sini al.  
`src/App.jsx` içindeki `COLAB_URL` değişkenine yapıştır.

## Yeni Sürüm Yayınlama

```bash
git tag v3.1.0
git push origin v3.1.0
# GitHub Actions otomatik build eder ve Release oluşturur
```

## Mimari

```
geopinn-frontend/        ← Bu repo
  src/App.jsx            ← React frontend
  main.cjs               ← Electron main process
  scripts/build-backend.cjs

geopinn-backend/         ← Python backend
  server.py              ← FastAPI
  engines/               ← Fizik motorları
    gravity_prism.py
    magnetic_prism.py
    csamt_1d.py
    ...
```

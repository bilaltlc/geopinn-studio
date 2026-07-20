// scripts/build-backend.cjs
//
// PyInstaller ile server.py -> server.exe derleyip, electron-builder'ın
// extraResources ayarının beklediği "backend/server/server.exe" konumuna
// otomatik kopyalar. Bu adımı elle yapıp unutmak yerine artık npm run dist
// öncesinde (predist) otomatik çalışır.
//
// Kullanım:
//   node scripts/build-backend.cjs
//
// Ortam değişkenleriyle özelleştirme:
//   SERVER_PY_PATH   -> server.py dosyasının yolu (varsayılan: ../server.py)
//   BACKEND_OUT_DIR  -> hedef klasör (varsayılan: <proje kökü>/backend/server)
//   PYTHON_BIN       -> python çalıştırılabilir adı/yolu (varsayılan: Windows'ta "python", diğerlerinde "python3")
//   EXCLUDE_MODULES  -> virgülle ayrılmış, analiz dışı bırakılacak modüller
//                       (varsayılan: "tensorflow" — server.py bunu kullanmıyor ama
//                       bazı Python kurulumlarında bozuk/eksik bir tensorflow paketi
//                       PyInstaller'ın hook taramasını çökertebiliyor)

const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const SERVER_PY = process.env.SERVER_PY_PATH || path.join(PROJECT_ROOT, 'server.py');
const BACKEND_OUT_DIR = process.env.BACKEND_OUT_DIR || path.join(PROJECT_ROOT, 'backend', 'server');
const PYINSTALLER_WORK = path.join(PROJECT_ROOT, '.pyinstaller-build');
const PYINSTALLER_DIST = path.join(PYINSTALLER_WORK, 'dist');
const DEFAULT_PYTHON = process.platform === 'win32' ? 'python' : 'python3';
const PYTHON_BIN = process.env.PYTHON_BIN || DEFAULT_PYTHON;
const EXCLUDE_MODULES = (process.env.EXCLUDE_MODULES || 'tensorflow')
  .split(',')
  .map((m) => m.trim())
  .filter(Boolean);

function fail(msg) {
  console.error(`\n[build-backend] HATA: ${msg}\n`);
  process.exit(1);
}

function main() {
  if (!fs.existsSync(SERVER_PY)) {
    fail(
      `server.py bulunamadı: ${SERVER_PY}\n` +
      `SERVER_PY_PATH ortam değişkeniyle doğru yolu belirtin, örn:\n` +
      `  SERVER_PY_PATH=./backend-src/server.py node scripts/build-backend.cjs`
    );
  }

  const excludeArgs = EXCLUDE_MODULES.flatMap((m) => ['--exclude-module', m]);
  if (excludeArgs.length) {
    console.log(`[build-backend] Analiz dışı bırakılan modüller: ${EXCLUDE_MODULES.join(', ')}`);
  }

  // "pyinstaller" komutu genelde pip'in Scripts/bin klasörü PATH'e eklenmediği
  // için bulunamıyor. "python -m PyInstaller" ise Python modül olarak
  // kurulduğu sürece PATH'ten bağımsız çalışır — bu yüzden onu tercih ediyoruz.
  console.log(`[build-backend] "${PYTHON_BIN} -m PyInstaller" ile derleniyor: ${SERVER_PY}`);
  const result = spawnSync(
    PYTHON_BIN,
    [
      '-m', 'PyInstaller',
      '--onefile',
      '--name', 'server',
      '--distpath', PYINSTALLER_DIST,
      '--workpath', path.join(PYINSTALLER_WORK, 'build'),
      '--specpath', PYINSTALLER_WORK,
      '--noconfirm',
      ...excludeArgs,
      SERVER_PY,
    ],
    { stdio: 'inherit', shell: process.platform === 'win32' }
  );

  if (result.error) {
    fail(
      `"${PYTHON_BIN}" çalıştırılamadı (${result.error.message}).\n` +
      `PYTHON_BIN ortam değişkeniyle doğru python yolunu belirtin, örn:\n` +
      `  PYTHON_BIN="C:/Users/telci/AppData/Local/Programs/Python/Python312/python.exe"`
    );
  }
  if (result.status !== 0) {
    fail(
      `PyInstaller derlemesi başarısız oldu (exit code ${result.status}).\n` +
      `Olası nedenler:\n` +
      `  1) PyInstaller kurulu değil -> ${PYTHON_BIN} -m pip install pyinstaller\n` +
      `  2) Bozuk bir bağımlılık paketi (örn. tensorflow) analiz sırasında hata veriyor ->\n` +
      `     EXCLUDE_MODULES ortam değişkenine o paketin adını ekleyin (virgülle ayırarak).`
    );
  }

  const exeName = process.platform === 'win32' ? 'server.exe' : 'server';
  const builtExe = path.join(PYINSTALLER_DIST, exeName);

  if (!fs.existsSync(builtExe)) {
    fail(`Derleme bitti ama beklenen çıktı yok: ${builtExe}`);
  }

  fs.mkdirSync(BACKEND_OUT_DIR, { recursive: true });
  const dest = path.join(BACKEND_OUT_DIR, exeName);
  fs.copyFileSync(builtExe, dest);

  console.log(`[build-backend] Kopyalandı: ${builtExe}\n                -> ${dest}`);
  console.log('[build-backend] Tamamlandı. "npm run dist" artık bu exe\'yi pakete gömecek.');
}

main();

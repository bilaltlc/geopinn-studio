'use strict';

const { execSync } = require('child_process');
const path = require('path');
const fs   = require('fs');

const serverPyPath = process.env.SERVER_PY_PATH
  || path.resolve(__dirname, '..', 'geopinn-backend', 'server.py');

if (!fs.existsSync(serverPyPath)) {
  console.error('[build-backend] server.py bulunamadı:', serverPyPath);
  process.exit(1);
}

const serverDir  = path.dirname(path.resolve(serverPyPath));
const enginesDir = path.join(serverDir, 'engines');
const outDir     = path.resolve(__dirname, '..', 'backend');
fs.mkdirSync(outDir, { recursive: true });

console.log('[build-backend] serverDir :', serverDir);
console.log('[build-backend] enginesDir:', enginesDir);
console.log('[build-backend] outDir    :', outDir);

if (!fs.existsSync(enginesDir)) {
  console.error('[build-backend] engines/ klasörü bulunamadı:', enginesDir);
  process.exit(1);
}

function findPython() {
  for (const bin of ['python', 'python3', 'py']) {
    try {
      const v = execSync(`${bin} --version 2>&1`, { stdio: 'pipe' }).toString();
      if (v.includes('Python 3')) return bin;
    } catch (_) {}
  }
  return null;
}

const py = findPython();
if (!py) { console.error('[build-backend] Python 3 bulunamadı.'); process.exit(1); }
console.log('[build-backend] Python:', py);

try { execSync(`${py} -m PyInstaller --version`, { stdio: 'pipe' }); }
catch (_) { execSync(`${py} -m pip install pyinstaller`, { stdio: 'inherit' }); }

const hidden = [
  'engines.gravity_prism', 'engines.magnetic_prism', 'engines.csamt_1d',
  'engines.harmonica_validation', 'engines.fvm_core',
  'engines.gravity_fvm', 'engines.magnetic_fvm', 'engines.petrophysics',
  'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim',
  'scipy.ndimage', 'scipy.sparse', 'scipy.sparse.linalg', 'scipy.interpolate',
  'fastapi', 'uvicorn', 'uvicorn.logging',
  'uvicorn.loops', 'uvicorn.loops.auto',
  'uvicorn.protocols', 'uvicorn.protocols.http', 'uvicorn.protocols.http.auto',
  'uvicorn.lifespan', 'uvicorn.lifespan.on',
  'pydantic', 'numpy', 'harmonica',
].map(m => `--hidden-import=${m}`).join(' ');

// TF/Keras hook'ları çöktüğü için dışla
const excluded = [
  'tensorflow', 'tensorflow_core', 'keras',
  'tensorboard', 'tf2onnx', 'jax', 'flax',
].map(m => `--exclude-module=${m}`).join(' ');

const addData = `--add-data "${enginesDir};engines"`;

const cmd = [
  `${py} -m PyInstaller`,
  '--noconfirm',
  '--onedir',
  `--distpath "${outDir}"`,
  `--workpath "${path.join(outDir, '_build_tmp')}"`,
  `--specpath "${path.join(outDir, '_specs')}"`,
  '--name server',
  hidden,
  excluded,
  addData,
  `"${path.resolve(serverPyPath)}"`,
].join(' ');

console.log('\n[build-backend] Komut:\n', cmd, '\n');

try {
  execSync(cmd, { stdio: 'inherit', cwd: serverDir });
  console.log('\n[build-backend] ✓ Tamamlandı:', outDir);
} catch (e) {
  console.error('[build-backend] Başarısız:', e.message);
  process.exit(1);
}

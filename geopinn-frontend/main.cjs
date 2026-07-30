'use strict';

const { app, BrowserWindow, ipcMain, shell, Tray, Menu, nativeImage } = require('electron');
const path  = require('path');
const fs    = require('fs');
const http  = require('http');
const { spawn } = require('child_process');

// ── Ayar dosyası (kullanıcı klasöründe, build'den bağımsız) ──────────────
const CONFIG_PATH = path.join(app.getPath('userData'), 'geopinn-config.json');

function loadConfig() {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      return JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
    }
  } catch(e) {}
  return { colabUrl: '', backendMode: 'local' };
}

function saveConfig(cfg) {
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2), 'utf8');
}

// ── Konfigürasyon ──────────────────────────────────────────────────────────
const BACKEND_PORT = 8000;
const IS_DEV       = !app.isPackaged;

const FRONTEND_DIST = app.isPackaged
  ? path.join(process.resourcesPath, 'frontend-dist')
  : path.join(__dirname, 'dist');

const BACKEND_EXE = app.isPackaged
  ? path.join(process.resourcesPath, 'backend', 'server', 'server.exe')
  : null;

let mainWindow  = null;
let backendProc = null;
let tray        = null;

// ── Backend başlatma ───────────────────────────────────────────────────────
function startBackend(cfg) {
  if (cfg.backendMode === 'colab' && cfg.colabUrl) {
    console.log('[main] Colab modu — yerel backend başlatılmıyor.');
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    let proc;
    if (BACKEND_EXE && fs.existsSync(BACKEND_EXE)) {
      proc = spawn(BACKEND_EXE,
        ['--host', '127.0.0.1', '--port', String(BACKEND_PORT)],
        { cwd: path.dirname(BACKEND_EXE), stdio: 'pipe' });
    } else {
      const serverPy = path.join(__dirname, '..', 'geopinn-backend', 'server.py');
      if (!fs.existsSync(serverPy)) { return resolve(); }
      proc = spawn('python', ['-m', 'uvicorn', 'server:app',
        '--host', '127.0.0.1', '--port', String(BACKEND_PORT)],
        { cwd: path.dirname(serverPy), stdio: 'pipe' });
    }
    backendProc = proc;
    proc.stdout?.on('data', d => console.log('[backend]', d.toString().trim()));
    proc.stderr?.on('data', d => console.error('[backend]', d.toString().trim()));

    const deadline = Date.now() + 30000;
    const poll = () => {
      http.get(`http://127.0.0.1:${BACKEND_PORT}/api/health`, res => {
        if (res.statusCode === 200) resolve();
        else if (Date.now() < deadline) setTimeout(poll, 500);
        else resolve();   // hata olsa da devam et
      }).on('error', () => {
        if (Date.now() < deadline) setTimeout(poll, 500);
        else resolve();
      });
    };
    setTimeout(poll, 800);
  });
}

// ── IPC handler'ları ───────────────────────────────────────────────────────
ipcMain.handle('get-config',    ()      => loadConfig());
ipcMain.handle('save-config',   (_, cfg) => { saveConfig(cfg); return true; });
ipcMain.handle('get-version',   ()      => app.getVersion());
ipcMain.handle('window-minimize',  () => mainWindow?.minimize());
ipcMain.handle('window-maximize',  () => {
  if (!mainWindow) return;
  mainWindow.isMaximized() ? mainWindow.restore() : mainWindow.maximize();
});
ipcMain.handle('window-hide',      () => mainWindow?.hide());
ipcMain.handle('window-close',     () => {
  if (tray) mainWindow?.hide(); else app.exit(0);
});
ipcMain.handle('window-is-max',    () => mainWindow?.isMaximized() ?? false);

ipcMain.handle('restart-backend', async () => {
  if (backendProc) { backendProc.kill(); backendProc = null; }
  const cfg = loadConfig();
  await startBackend(cfg);
  return true;
});

// ── Pencere ────────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440, height: 920, minWidth: 960, minHeight: 640,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
    frame: true,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    show:  false,
    backgroundColor: '#0A0C0F',
    icon: path.join(__dirname, 'build', 'icon.ico'),
    title: 'GeoPINN Studio 3.0',
  });

  // Tam ekran — uygulama açılışında maximize
  mainWindow.once('ready-to-show', () => {
    mainWindow.maximize();
    mainWindow.show();
  });

  // Kapatma → tepside sakla (Windows/Linux)
  mainWindow.on('close', (e) => {
    if (process.platform !== 'darwin' && tray) {
      e.preventDefault();
      mainWindow.hide();
    }
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url); return { action: 'deny' };
  });

  if (IS_DEV) {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(FRONTEND_DIST, 'index.html'));
  }
  mainWindow.on('closed', () => { mainWindow = null; });
}

app.whenReady().then(async () => {
  const cfg = loadConfig();
  await startBackend(cfg);
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
    else if (mainWindow) { mainWindow.show(); mainWindow.focus(); }
  });

  // ── Sistem tepsisi ─────────────────────────────────────────────────────────
  try {
    const iconPath = path.join(__dirname, 'build', 'icon.ico');
    const trayIcon = fs.existsSync(iconPath)
      ? nativeImage.createFromPath(iconPath).resize({ width:16, height:16 })
      : nativeImage.createEmpty();

    tray = new Tray(trayIcon);
    tray.setToolTip('GeoPINN Studio 3.0');

    const contextMenu = Menu.buildFromTemplate([
      { label: 'GeoPINN Studio', enabled: false },
      { type: 'separator' },
      { label: 'Pencereyi Göster', click: () => {
          if (mainWindow) { mainWindow.show(); mainWindow.focus(); mainWindow.restore(); }
      }},
      { label: 'Tam Ekran', click: () => {
          if (mainWindow) { mainWindow.show(); mainWindow.maximize(); }
      }},
      { type: 'separator' },
      { label: 'Çıkış', click: () => {
          tray = null;
          if (backendProc) backendProc.kill();
          app.exit(0);
      }},
    ]);

    tray.setContextMenu(contextMenu);
    // Tek tıkla pencereyi göster/gizle
    tray.on('click', () => {
      if (!mainWindow) return;
      if (mainWindow.isVisible() && mainWindow.isFocused()) mainWindow.hide();
      else { mainWindow.show(); mainWindow.focus(); }
    });
  } catch(e) {
    console.warn('[tray] Sistem tepsisi oluşturulamadı:', e.message);
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (backendProc) backendProc.kill();
    app.quit();
  }
});

app.on('before-quit', () => {
  if (backendProc) { backendProc.kill(); backendProc = null; }
});

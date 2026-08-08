'use strict';

const { app, BrowserWindow, ipcMain, shell, Menu, Tray, nativeImage, screen } = require('electron');
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

let mainWindow   = null;
let detachWindow = null;
let backendProc  = null;
let tray         = null;

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
ipcMain.handle('restart-backend', async () => {
  if (backendProc) { backendProc.kill(); backendProc = null; }
  const cfg = loadConfig();
  await startBackend(cfg);
  return true;
});

// Pencere kontrol IPC'leri
ipcMain.handle('window-minimize',  (event) => {
  const senderWin = BrowserWindow.fromWebContents(event.sender);
  (senderWin || mainWindow)?.minimize();
});
ipcMain.handle('window-maximize',  (event) => {
  const senderWin = BrowserWindow.fromWebContents(event.sender) || mainWindow;
  if (!senderWin) return;
  senderWin.isMaximized() ? senderWin.restore() : senderWin.maximize();
});
ipcMain.handle('window-is-max',   () => mainWindow?.isMaximized() ?? false);
ipcMain.handle('window-close',    (event) => {
  // Hangi pencereden geldiğini bul
  const senderWin = BrowserWindow.fromWebContents(event.sender);
  // mainWindow ise tray'e küçült veya çık
  if (!senderWin || senderWin === mainWindow) {
    if (tray) mainWindow?.hide();
    else app.exit(0);
    return;
  }
  // Panel penceresi ise sadece onu kapat
  senderWin.close();
});

// 3D detach penceresi
ipcMain.handle('open-detach-window', (_, url) => { createDetachWindow(url); return true; });
ipcMain.handle('close-detach-window', () => { detachWindow?.close(); return true; });

// Harici monitöre taşı
// Panel penceresi — mevcut ekranda veya harici monitörde açılır, sürüklenebilir
const panelWindows = new Map();

ipcMain.handle('open-panel-window', (_, {panelId, x, y, w, h}) => {
  // Zaten açıksa öne getir
  if (panelWindows.has(panelId)) {
    const existing = panelWindows.get(panelId);
    if (!existing.isDestroyed()) { existing.focus(); return true; }
  }

  const win = new BrowserWindow({
    x: Math.max(0, x || 300),
    y: Math.max(0, y || 200),
    width:  w || 600,
    height: h || 500,
    minWidth: 300, minHeight: 200,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
    frame: false,
    backgroundColor: '#0A0C0F',
    title: 'GeoPINN Studio — Panel',
    icon: path.join(__dirname, 'build', 'icon.ico'),
    // Bağımsız pencere — ana pencere kapanınca bu da kapanır
    parent: null,
    skipTaskbar: false,
    alwaysOnTop: false,
  });

  Menu.setApplicationMenu(null);

  // Aynı frontend — panel parametresiyle
  if (IS_DEV) {
    win.loadURL(`http://localhost:5173/?panel=${panelId}`);
  } else {
    win.loadFile(path.join(FRONTEND_DIST, 'index.html'), { hash: `panel=${panelId}` });
  }

  panelWindows.set(panelId, win);
  win.on('close', (e) => {
    // Panel kapanınca sadece kendini kapat, ana pencereye dokunma
    panelWindows.delete(panelId);
  });
  return true;
});

ipcMain.handle('move-to-external-display', (_, {x,y,w,h}) => {
  const displays = screen.getAllDisplays();
  const primary  = screen.getPrimaryDisplay();
  const external = displays.find(d=>d.id!==primary.id);
  if (!external) { console.log('[main] Harici monitör bulunamadı'); return false; }
  const win = BrowserWindow.getFocusedWindow() || mainWindow;
  if (!win) return false;
  const ex = external.workArea;
  // Boyutu koru, harici monitörün ortasına taşı
  const nw = Math.min(w, ex.width);
  const nh = Math.min(h, ex.height);
  win.setBounds({
    x: ex.x + Math.round((ex.width-nw)/2),
    y: ex.y + Math.round((ex.height-nh)/2),
    width: nw, height: nh,
  });
  return true;
});

// ── Menüyü tamamen kaldır ─────────────────────────────────────────────────
Menu.setApplicationMenu(null);

// ── Pencere ────────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440, height: 920, minWidth: 960, minHeight: 640,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
    frame: false,
    show:  false,
    backgroundColor: '#0A0C0F',
    icon: path.join(__dirname, 'build', 'icon.ico'),
    title: 'GeoPINN Studio 3.0',
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.maximize();
    mainWindow.show();
  });

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url); return { action: 'deny' };
  });

  // Kapatınca tepside sakla
  mainWindow.on('close', (e) => {
    if (tray) { e.preventDefault(); mainWindow.hide(); }
  });

  if (IS_DEV) {
    mainWindow.loadURL('http://localhost:5173');
  } else {
    mainWindow.loadFile(path.join(FRONTEND_DIST, 'index.html'));
  }
  mainWindow.on('closed', () => { mainWindow = null; });
}

// ── 3D Detach penceresi ────────────────────────────────────────────────────
function createDetachWindow(url) {
  if (detachWindow && !detachWindow.isDestroyed()) {
    detachWindow.focus(); return;
  }
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  detachWindow = new BrowserWindow({
    width: Math.round(width * 0.7),
    height: Math.round(height * 0.8),
    minWidth: 640, minHeight: 480,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.cjs'),
    },
    frame: false,
    backgroundColor: '#0A0C0F',
    title: 'GeoPINN Studio — 3D Viewer',
    icon: path.join(__dirname, 'build', 'icon.ico'),
    parent: mainWindow,
  });
  Menu.setApplicationMenu(null);
  // Detach penceresi aynı URL'yi yükler, ?detach=1 param ile 3D moduna geçer
  if (IS_DEV) {
    detachWindow.loadURL('http://localhost:5173?detach=1');
  } else {
    detachWindow.loadFile(path.join(FRONTEND_DIST, 'index.html'), { query: { detach: '1' } });
  }
  detachWindow.on('closed', () => { detachWindow = null; });
}

app.whenReady().then(async () => {
  const cfg = loadConfig();
  await startBackend(cfg);
  createWindow();

  // Sistem tepsisi
  try {
    const iconPath = path.join(__dirname, 'build', 'icon.ico');
    const trayIcon = fs.existsSync(iconPath)
      ? nativeImage.createFromPath(iconPath).resize({ width:16, height:16 })
      : nativeImage.createEmpty();
    tray = new Tray(trayIcon);
    tray.setToolTip('GeoPINN Studio 3.0');
    const ctxMenu = Menu.buildFromTemplate([
      { label:'GeoPINN Studio 3.0', enabled:false },
      { type:'separator' },
      { label:'Göster', click:()=>{ mainWindow?.show(); mainWindow?.focus(); }},
      { label:'Tam Ekran', click:()=>{ mainWindow?.show(); mainWindow?.maximize(); }},
      { type:'separator' },
      { label:'Çıkış', click:()=>{ tray=null; backendProc?.kill(); app.exit(0); }},
    ]);
    tray.setContextMenu(ctxMenu);
    tray.on('click', ()=>{
      if (!mainWindow) return;
      if (mainWindow.isVisible() && mainWindow.isFocused()) mainWindow.hide();
      else { mainWindow.show(); mainWindow.focus(); }
    });
  } catch(e) { console.warn('[tray] oluşturulamadı:', e.message); }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
    else mainWindow?.show();
  });
});

app.on('window-all-closed', () => {
  // Panel pencereleri veya tray modu varsa çıkma
  if (tray) return;  // tray varsa uygulama arka planda devam eder
  if (panelWindows.size > 0) return;  // panel pencereleri açıksa çıkma
  if (process.platform !== 'darwin') {
    if (backendProc) backendProc.kill();
    app.quit();
  }
});

app.on('before-quit', () => {
  if (backendProc) { backendProc.kill(); backendProc = null; }
});

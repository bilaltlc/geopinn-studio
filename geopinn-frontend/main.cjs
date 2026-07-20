const { app, BrowserWindow, session, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let backendProcess = null;
let mainWindow;

function getBackendPath() {
  // Geliştirmede vs paketlenmiş uygulamada exe yolu farklı olur
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'backend', 'server', 'server.exe');
  }
  return path.join(__dirname, 'resources', 'backend', 'server', 'server.exe');
}

function startBackend() {
  const backendPath = getBackendPath();

  // Kopyalama adımı atlanınca (backend/server/server.exe eksikse) spawn
  // sessizce başarısız oluyordu; windowsHide yüzünden hiçbir konsolda
  // görünmüyordu. Artık eksikse kullanıcıya görünür bir hata gösteriyoruz.
  if (!fs.existsSync(backendPath)) {
    const msg = `Backend çalıştırılabilir dosyası bulunamadı:\n${backendPath}\n\n` +
      `PyInstaller çıktısı (server.exe) "backend/server/" klasörüne kopyalanmamış olabilir.\n` +
      `"npm run predist" (scripts/build-backend.cjs) betiğini çalıştırıp tekrar paketleyin.`;
    console.error('[backend] ' + msg);
    dialog.showErrorBox('Backend Başlatılamadı', msg);
    return;
  }

  backendProcess = spawn(backendPath, [], {
    cwd: path.dirname(backendPath),
    windowsHide: true, // arka planda çalışsın, konsol penceresi açılmasın
  });

  backendProcess.stdout.on('data', (data) => {
    console.log(`[backend] ${data}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`[backend-err] ${data}`);
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
    backendProcess = null;
  });

  backendProcess.on('error', (err) => {
    console.error('Backend process başlatılamadı:', err);
    dialog.showErrorBox('Backend Başlatılamadı', `${backendPath}\n\n${err.message}`);
  });
}

function stopBackend() {
  if (backendProcess) {
    backendProcess.kill(); // Windows'ta SIGTERM eşdeğeri
    backendProcess = null;
  }
}

function getFrontendIndexPath() {
  // Paketlenmiş uygulamada frontend, extraResources ile 'frontend-dist' klasörüne kopyalanıyor
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'frontend-dist', 'index.html');
  }
  // Geliştirmede Vite'in build çıktısı olan 'dist' klasörünü kullan
  return path.join(__dirname, 'dist', 'index.html');
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      webSecurity: false,
      nodeIntegration: false,
      contextIsolation: true
    }
  });
  // loadURL ve loadFile art arda çağrılıyordu; loadFile her zaman loadURL'i
  // geçersiz kılıyordu. Artık paketlenme durumuna göre yalnızca biri çalışıyor.
  if (app.isPackaged) {
    mainWindow.loadFile(getFrontendIndexPath());
  } else {
    mainWindow.loadURL('http://localhost:5173');
  }
}

app.whenReady().then(() => {
  startBackend();
  createWindow();
});

app.on('before-quit', () => {
  stopBackend();
});

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') app.quit();
});
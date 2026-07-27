'use strict';
const { contextBridge, ipcRenderer } = require('electron');

// Frontend'in Electron API'lerine güvenli erişimi
contextBridge.exposeInMainWorld('electronAPI', {
  getConfig:       ()    => ipcRenderer.invoke('get-config'),
  saveConfig:      (cfg) => ipcRenderer.invoke('save-config', cfg),
  getVersion:      ()    => ipcRenderer.invoke('get-version'),
  restartBackend:  ()    => ipcRenderer.invoke('restart-backend'),
  isElectron:      true,
});

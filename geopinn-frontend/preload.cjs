'use strict';
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Config
  getConfig:          ()      => ipcRenderer.invoke('get-config'),
  saveConfig:         (cfg)   => ipcRenderer.invoke('save-config', cfg),
  getVersion:         ()      => ipcRenderer.invoke('get-version'),
  restartBackend:     ()      => ipcRenderer.invoke('restart-backend'),
  isElectron:         true,

  // Pencere kontrol
  minimize:           ()      => ipcRenderer.invoke('window-minimize'),
  maximize:           ()      => ipcRenderer.invoke('window-maximize'),
  isMaximized:        ()      => ipcRenderer.invoke('window-is-max'),
  close:              ()      => ipcRenderer.invoke('window-close'),

  // 3D detach penceresi
  openDetachWindow:   (url)   => ipcRenderer.invoke('open-detach-window', url),
  closeDetachWindow:      ()      => ipcRenderer.invoke('close-detach-window'),
  moveToExternalDisplay:  (opts)  => ipcRenderer.invoke('move-to-external-display', opts),
  openPanelWindow:        (opts)  => ipcRenderer.invoke('open-panel-window', opts),

  // Panel penceresi için pencere kontrolleri
  panelMinimize: () => ipcRenderer.invoke('window-minimize'),
  panelClose:    () => ipcRenderer.invoke('window-close'),

  // URL'deki panel parametresini oku
  getPanelId: () => new URLSearchParams(window.location.search).get('panel'),
});

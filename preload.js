const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    getLocalIp: () => ipcRenderer.invoke('get-local-ip'),
    startInstance: (config) => ipcRenderer.invoke('start-instance', config),
    stopInstance: (id) => ipcRenderer.invoke('stop-instance', id),
    onInstanceLog: (callback) => ipcRenderer.on('instance-log', (event, value) => callback(value)),
    onInstanceStopped: (callback) => ipcRenderer.on('instance-stopped', (event, value) => callback(value))
});

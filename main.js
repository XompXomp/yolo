const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const os = require('os');
const fs = require('fs');

let mainWindow;
const instances = new Map();
const modelsPath = path.join(__dirname, 'models');

// Ensure models directory exists
if (!fs.existsSync(modelsPath)) {
    fs.mkdirSync(modelsPath);
}


function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1000,
        height: 700,
        titleBarStyle: 'hiddenInset', // Premium macOS look
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
        },
    });

    mainWindow.loadFile('index.html');
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    // Kill all python processes on exit
    for (const [id, proc] of instances) {
        proc.kill();
    }
    if (process.platform !== 'darwin') app.quit();
});

// Helper to get local IP
function getLocalIp() {
    const interfaces = os.networkInterfaces();
    for (const name of Object.keys(interfaces)) {
        for (const iface of interfaces[name]) {
            if (iface.family === 'IPv4' && !iface.internal) {
                return iface.address;
            }
        }
    }
    return 'localhost';
}

// IPC Handlers
ipcMain.handle('get-local-ip', () => getLocalIp());

ipcMain.handle('get-models', () => {
    try {
        const files = fs.readdirSync(modelsPath);
        return files.filter(file => file.endsWith('.pt') || file.endsWith('.onnx'));
    } catch (err) {
        console.error("Failed to read models directory", err);
        return [];
    }
});

ipcMain.handle('start-instance', async (event, config) => {

    const { id, port, model, width, height } = config;

    // Decide python command (python3 for Mac)
    const pythonCmd = process.platform === 'darwin' ? 'python3' : 'python';
    const scriptPath = path.join(__dirname, 'yolo', 'yolo.py');

    const args = [
        scriptPath,
        '--model', model || 'yolov8n.pt',
        '--port', port.toString(),
        '--name', id,
        '--device', 'mps'
    ];

    if (width) args.push('--width', width.toString());
    if (height) args.push('--height', height.toString());

    const proc = spawn(pythonCmd, args);

    instances.set(id, proc);

    proc.stdout.on('data', (data) => {
        console.log(`[${id}] ${data}`);
        mainWindow.webContents.send('instance-log', { id, data: data.toString() });
    });

    proc.stderr.on('data', (data) => {
        console.error(`[${id}] ERR: ${data}`);
        mainWindow.webContents.send('instance-log', { id, data: data.toString(), isError: true });
    });

    proc.on('close', (code) => {
        console.log(`[${id}] exited with code ${code}`);
        instances.delete(id);
        mainWindow.webContents.send('instance-stopped', { id, code });
    });

    return { status: 'success', pid: proc.pid };
});

ipcMain.handle('stop-instance', async (event, id) => {
    const proc = instances.get(id);
    if (proc) {
        proc.kill();
        instances.delete(id);
        return { status: 'stopped' };
    }
    return { status: 'not-running' };
});

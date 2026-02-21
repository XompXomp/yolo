const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const os = require('os');
const fs = require('fs');

let mainWindow;
const instances = new Map();
const modelsPath = path.join(__dirname, 'models');

// Load Balancer State
const activeModels = new Map(); // modelName -> { ports: [], nextIndex: 0 }

function registerInstance(model, port) {
    if (!activeModels.has(model)) {
        activeModels.set(model, { ports: [], nextIndex: 0 });
    }
    const modelState = activeModels.get(model);
    if (!modelState.ports.includes(port)) {
        modelState.ports.push(port);
        console.log(`[Overseer] Registered port ${port} for model ${model}. Total: ${modelState.ports.length}`);
    }
}

function unregisterInstance(model, port) {
    if (activeModels.has(model)) {
        const modelState = activeModels.get(model);
        modelState.ports = modelState.ports.filter(p => p !== port);
        console.log(`[Overseer] Unregistered port ${port} from model ${model}. Remaining: ${modelState.ports.length}`);
        if (modelState.ports.length === 0) {
            activeModels.delete(model);
        } else if (modelState.nextIndex >= modelState.ports.length) {
            modelState.nextIndex = 0; // Reset index if it's now out of bounds
        }
    }
}

// Ensure models directory exists
if (!fs.existsSync(modelsPath)) {
    fs.mkdirSync(modelsPath);
}

// ------------------------------------------------------------------
// OVERSEER: EXPRESS PROXY SERVER
// ------------------------------------------------------------------
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

const gateway = express();
const GATEWAY_PORT = 9000;

gateway.post('/detect', (req, res, next) => {
    // 1. Extract model from query param (e.g. /detect?model=yolov8n.pt)
    const requestedModel = req.query.model;

    if (!requestedModel) {
        return res.status(400).json({ error: "Missing 'model' query parameter" });
    }

    // 2. Look up active instances for this model
    const modelState = activeModels.get(requestedModel);

    if (!modelState || modelState.ports.length === 0) {
        return res.status(503).json({ error: `No active instances found for model: ${requestedModel}` });
    }

    // 3. Round-robin selection
    const ports = modelState.ports;
    const targetPort = ports[modelState.nextIndex];

    // Update index for next time
    modelState.nextIndex = (modelState.nextIndex + 1) % ports.length;

    console.log(`[Overseer] Routing request for ${requestedModel} to Port ${targetPort}`);

    // 4. Dynamically proxy the request to the selected port
    const dynamicProxy = createProxyMiddleware({
        target: `http://localhost:${targetPort}`,
        changeOrigin: true,
        logLevel: 'silent' // We handle our own logging
    });

    // Execute the proxy middleware
    dynamicProxy(req, res, next);
});

gateway.listen(GATEWAY_PORT, () => {
    console.log(`[Overseer] Load Balancer Gateway listening on http://localhost:${GATEWAY_PORT}`);
});
// ------------------------------------------------------------------


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
        proc.process.kill();
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
    const modelToRun = model || 'yolov8n.pt';

    // Decide python command (python3 for Mac)
    const pythonCmd = process.platform === 'darwin' ? 'python3' : 'python';
    const scriptPath = path.join(__dirname, 'yolo', 'yolo.py');

    const args = [
        scriptPath,
        '--model', modelToRun,
        '--port', port.toString(),
        '--name', id,
        '--device', 'mps'
    ];

    if (width) args.push('--width', width.toString());
    if (height) args.push('--height', height.toString());

    const proc = spawn(pythonCmd, args);

    // Save model and port so we know what to unregister later
    instances.set(id, { process: proc, model: modelToRun, port: port });
    registerInstance(modelToRun, port);

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
        unregisterInstance(modelToRun, port);
        instances.delete(id);
        mainWindow.webContents.send('instance-stopped', { id, code });
    });

    return { status: 'success', pid: proc.pid };
});

ipcMain.handle('stop-instance', async (event, id) => {
    const instanceData = instances.get(id);
    if (instanceData) {
        instanceData.process.kill();
        unregisterInstance(instanceData.model, instanceData.port);
        instances.delete(id);
        return { status: 'stopped' };
    }
    return { status: 'not-running' };
});

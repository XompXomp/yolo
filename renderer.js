let localIp = 'localhost';
let instanceCounter = 1;
const instances = new Map();

// Elements
const addInstanceBtn = document.getElementById('add-instance-btn');
const modalOverlay = document.getElementById('modal-overlay');
const cancelAddBtn = document.getElementById('cancel-add');
const confirmAddBtn = document.getElementById('confirm-add');
const modelSelect = document.getElementById('model-select');
const instancesContainer = document.getElementById('instances-container');
const emptyState = document.getElementById('empty-state');
const localIpDisplay = document.getElementById('local-ip-display');

// Init
window.electronAPI.getLocalIp().then(ip => {
    localIp = ip;
    localIpDisplay.innerText = `IP: ${localIp}`;
});

// UI Event Handlers
addInstanceBtn.onclick = () => {
    modalOverlay.classList.remove('hidden');
};

cancelAddBtn.onclick = () => {
    modalOverlay.classList.add('hidden');
};

confirmAddBtn.onclick = async () => {
    const id = `yolo${instanceCounter++}`;
    const port = 9000 + instanceCounter; // Sequential ports
    const model = modelSelect.value;

    const config = { id, port, model };

    modalOverlay.classList.add('hidden');

    // Create UI Card
    createInstanceCard(config);

    // Call Main Process to start
    const result = await window.electronAPI.startInstance(config);
    console.log('Instance started:', result);
};

function createInstanceCard(config) {
    emptyState.classList.add('hidden');

    const { id, port, model } = config;
    const wsUrl = `ws://${localIp}:${port}/detections`;

    const card = document.createElement('div');
    card.className = 'instance-card';
    card.id = `card-${id}`;
    card.innerHTML = `
        <div class="instance-header">
            <div class="instance-info">
                <h3>${id}</h3>
                <span class="instance-tag">${model}</span>
            </div>
            <div class="status-indicator">
                <span class="status-dot running" id="dot-${id}"></span>
                <span id="status-text-${id}">Running</span>
            </div>
        </div>
        
        <div class="instance-details">
            <div class="detail-item">
                <span>Port</span>
                <span class="detail-value">${port}</span>
            </div>
            <div class="detail-item">
                <span>WebSocket</span>
                <span class="detail-value">${wsUrl}</span>
            </div>
        </div>
        
        <div class="instance-actions">
            <button class="btn btn-secondary btn-sm" onclick="stopInstance('${id}')">Stop</button>
        </div>
    `;

    instancesContainer.appendChild(card);
    instances.set(id, { ...config, card });
}

window.stopInstance = async (id) => {
    await window.electronAPI.stopInstance(id);
};

// IPC Listeners
window.electronAPI.onInstanceStopped(({ id, code }) => {
    const dot = document.getElementById(`dot-${id}`);
    const text = document.getElementById(`status-text-${id}`);
    if (dot) dot.className = 'status-dot stopped';
    if (text) text.innerText = 'Stopped';
});

window.electronAPI.onInstanceLog(({ id, data, isError }) => {
    // Optionally log to a small console in the UI
    console.log(`[${id}] ${data}`);
});

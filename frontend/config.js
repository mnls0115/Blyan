// API Configuration
const API_CONFIG = {
    // 🔧 CONFIGURATION FOR RUNPOD GPU NODE

    // Option 1: Direct RunPod IP (replace YOUR_RUNPOD_IP with actual IP)
    baseURL: 'http://YOUR_RUNPOD_IP:8002',

    // Option 2: Environment variable (set RUNPOD_IP in browser console)
    // baseURL: window.RUNPOD_IP ? `http://${window.RUNPOD_IP}:8002` : 'http://127.0.0.1:8000',

    // Option 3: Auto-detect from current domain
    // baseURL: `${window.location.protocol}//${window.location.hostname}:8002`,

    // Option 4: Local development (uncomment for local testing)
    // baseURL: 'http://127.0.0.1:8002',

    // Individual endpoints
    balance: '/balance/',
    chain: '/chain/B/blocks?limit=10',
    polStatus: '/pol/status'
};

// Export for use in other files
window.API_CONFIG = API_CONFIG;
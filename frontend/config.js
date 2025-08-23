// API Configuration
const API_CONFIG = {
    // 🔧 CONFIGURATION FOR API ENDPOINTS

    // Use main server API (Digital Ocean - HTTPS enabled)
    baseURL: 'https://blyan.com/api',

    // Alternative configurations:
    // Option 1: Direct GPU node (requires HTTPS proxy or local dev)
    // baseURL: 'https://your-gpu-node.com:8002',

    // Option 2: Local development (browser must allow mixed content)
    // baseURL: 'http://127.0.0.1:8002',

    // Option 3: Environment variable
    // baseURL: window.API_URL || 'https://blyan.com/api',

    // Individual endpoints
    balance: '/balance/',
    chain: '/chain/B/blocks?limit=10',
    polStatus: '/pol/status'
};

// Export for use in other files
window.API_CONFIG = API_CONFIG;
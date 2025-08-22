// API Configuration
const API_CONFIG = {
    // 🔧 CONFIGURATION OPTIONS - Choose one:

    // Option 1: Auto-detect GPU node (if accessing via port 8002)
    // baseURL: window.location.port === '8002' ? 'http://127.0.0.1:8002' : 'http://127.0.0.1:8000',

    // Option 2: Always use GPU node (recommended for testing)
    baseURL: 'http://127.0.0.1:8002',

    // Option 3: Always use main node
    // baseURL: 'http://127.0.0.1:8000',

    // Option 4: Environment-based (set NODE_URL in browser console)
    // baseURL: window.NODE_URL || 'http://127.0.0.1:8000',

    // Individual endpoints
    balance: '/balance/',
    chain: '/chain/B/blocks?limit=10',
    polStatus: '/pol/status'
};

// Export for use in other files
window.API_CONFIG = API_CONFIG;
// API Configuration
const API_CONFIG = {
    // 🔧 CONFIGURATION FOR API ENDPOINTS

    // Use main server API (Digital Ocean - HTTPS enabled)
    // ⚠️  IMPORTANT: Always use HTTPS in production to avoid Mixed Content errors
    baseURL: 'https://blyan.com/api',

    // Alternative configurations (for development only):
    // Option 1: Direct GPU node (requires HTTPS proxy or local dev)
    // baseURL: 'https://your-gpu-node.com:8002',

    // Option 2: Local development (browser must allow mixed content)
    // baseURL: 'http://127.0.0.1:8002',

    // Option 3: Environment variable (falls back to production HTTPS)
    // baseURL: window.API_URL || 'https://blyan.com/api',

    // Individual endpoints
    balance: '/balance/',
    chain: '/chain/B/blocks?limit=10',
    polStatus: '/pol/status',

    // Helper function to ensure HTTPS in production
    ensureHttps: function(url) {
        if (typeof window !== 'undefined' && window.location.protocol === 'https:') {
            return url.replace(/^http:/, 'https:');
        }
        return url;
    }
};

// Ensure API base URL uses HTTPS when served over HTTPS
API_CONFIG.baseURL = API_CONFIG.ensureHttps(API_CONFIG.baseURL);

// Export for use in other files
window.API_CONFIG = API_CONFIG;
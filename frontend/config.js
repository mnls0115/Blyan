// API Configuration
const API_CONFIG = {
    // Use environment variable or fallback to localhost
    baseURL: (window.location.protocol === 'file:' || 
              window.location.hostname === 'localhost' || 
              window.location.hostname === '127.0.0.1' ||
              window.location.hostname === '') 
        ? 'http://127.0.0.1:8000' 
        : `${window.location.protocol}//${window.location.hostname}/api`,
    
    // Individual endpoints
    chat: '/chat',
    balance: '/balance/',
    chain: '/chain/B/blocks?limit=10',
    polStatus: '/pol/status'
};

// Export for use in other files
window.API_CONFIG = API_CONFIG;
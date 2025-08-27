// API Configuration
const API_CONFIG = {
    // 🔧 CONFIGURATION FOR API ENDPOINTS

    // Auto-detect: Use local API if available, otherwise production
    // This prevents rate limiting issues during development
    baseURL: (function() {
        // Check URL parameters first (allows override)
        const urlParams = new URLSearchParams(window.location.search);
        const apiUrl = urlParams.get('api');
        if (apiUrl) {
            console.log('🎯 Using API from URL param:', apiUrl);
            return apiUrl;
        }
        
        // Check if we're in local development
        const isLocalhost = window.location.hostname === 'localhost' || 
                          window.location.hostname === '127.0.0.1' ||
                          window.location.hostname.startsWith('192.168.');
        
        if (isLocalhost) {
            // Use local GPU node API
            console.log('🏠 Using local API (http://127.0.0.1:8000)');
            return 'http://127.0.0.1:8000';
        } else {
            // Use appropriate endpoint based on protocol
            if (window.location.protocol === 'https:') {
                console.log('🌐 Using DigitalOcean service node via HTTPS proxy');
                return 'https://blyan.com/api';
            } else {
                console.log('🌐 Using DigitalOcean service node direct HTTP');
                return 'http://165.227.221.225:8000';
            }
        }
    })(),

    // Manual override options:
    // baseURL: 'http://127.0.0.1:8000',  // Force local
    // baseURL: 'https://blyan.com/api',  // Force production

    // Individual endpoints
    balance: '/balance/',
    chain: '/chain/B/blocks?limit=10',
    polStatus: '/pol/status',
    health: '/health',  // Basic health check endpoint

    // Helper function to ensure HTTPS in production
    // DISABLED: DigitalOcean service node uses HTTP only (no SSL cert for IP)
    ensureHttps: function(url) {
        // Always return URL as-is, don't convert to HTTPS
        return url;
    }
};

// Do NOT force HTTPS for API URLs - the API server runs on HTTP
// API_CONFIG.baseURL = API_CONFIG.ensureHttps(API_CONFIG.baseURL);

// Export for use in other files
window.API_CONFIG = API_CONFIG;
// GPU Node Configuration for Chat Interface
// This config points the frontend to your RunPod GPU node

const API_CONFIG = {
    // Point to your GPU node instead of main server
    baseURL: (function() {
        // Your GPU node endpoint
        const GPU_NODE_URL = 'http://184.144.239.170:8000';
        
        // Check if we're in local development
        const isLocalhost = window.location.hostname === 'localhost' || 
                          window.location.hostname === '127.0.0.1';
        
        if (isLocalhost) {
            // If testing locally, still use GPU node
            console.log('🚀 Using GPU node API:', GPU_NODE_URL);
            return GPU_NODE_URL;
        } else {
            // In production, also use GPU node
            console.log('🚀 Using GPU node API:', GPU_NODE_URL);
            return GPU_NODE_URL;
        }
    })(),

    // Individual endpoints
    balance: '/balance/',
    chain: '/chain/B/blocks?limit=10',
    polStatus: '/pol/status',
    health: '/health',
    
    // Helper function to ensure HTTPS in production
    ensureHttps: function(url) {
        // Don't force HTTPS for GPU node (it's HTTP)
        return url;
    }
};

// Export for use in other files
window.API_CONFIG = API_CONFIG;

console.log('✅ GPU Node config loaded - API endpoint:', API_CONFIG.baseURL);
// Dynamic API Configuration
// Auto-discovers GPU nodes or uses environment-based configuration

const API_CONFIG = {
    // Dynamic baseURL selection
    baseURL: (function() {
        // Check URL parameters first (allows override)
        const urlParams = new URLSearchParams(window.location.search);
        const apiUrl = urlParams.get('api');
        if (apiUrl) {
            console.log('🎯 Using API from URL param:', apiUrl);
            return apiUrl;
        }
        
        // Check localStorage for saved GPU node
        const savedNode = localStorage.getItem('gpu_node_url');
        if (savedNode) {
            console.log('💾 Using saved GPU node:', savedNode);
            return savedNode;
        }
        
        // Check if we're in local development
        const isLocalhost = window.location.hostname === 'localhost' || 
                          window.location.hostname === '127.0.0.1';
        
        if (isLocalhost) {
            // Try local GPU node first
            console.log('🏠 Using local API (http://localhost:8000)');
            return 'http://localhost:8000';
        }
        
        // Default to main server (which can proxy to GPU nodes)
        console.log('🌐 Using main server API');
        return 'https://blyan.com/api';
    })(),

    // Allow runtime configuration
    setGPUNode: function(url) {
        localStorage.setItem('gpu_node_url', url);
        this.baseURL = url;
        console.log('✅ GPU node configured:', url);
        // Reload to apply
        window.location.reload();
    },
    
    // Clear custom configuration
    clearGPUNode: function() {
        localStorage.removeItem('gpu_node_url');
        window.location.reload();
    },

    // Individual endpoints
    balance: '/balance/',
    chain: '/chain/B/blocks?limit=10',
    polStatus: '/pol/status',
    health: '/health',
    
    // Helper function to ensure HTTPS in production
    ensureHttps: function(url) {
        if (typeof window !== 'undefined' && window.location.protocol === 'https:' && !url.includes('localhost') && !url.match(/\d+\.\d+\.\d+\.\d+/)) {
            return url.replace(/^http:/, 'https:');
        }
        return url;
    }
};

// Export for use in other files
window.API_CONFIG = API_CONFIG;

// Add helper message
console.log(`
🔧 Configuration Options:
1. Add ?api=http://your-gpu-node:8000 to URL
2. Run: API_CONFIG.setGPUNode('http://your-gpu-node:8000')
3. Let it auto-detect based on your environment
Current: ${API_CONFIG.baseURL}
`);
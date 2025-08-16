/**
 * API Authentication Management for Blyan
 * Handles API key lifecycle, refresh, and error recovery
 */

class APIAuthManager {
    constructor() {
        this.apiKey = null;
        this.isRefreshing = false;
        this.pendingRequests = [];
        this.maxRetries = 3;
        this.retryDelay = 1000; // 1 second
        
        // Load existing API key
        this.loadStoredApiKey();
    }

    /**
     * Load API key from localStorage
     */
    loadStoredApiKey() {
        try {
            const stored = localStorage.getItem('apiKey');
            const keyInfo = localStorage.getItem('apiKeyInfo');
            
            if (stored && keyInfo) {
                const info = JSON.parse(keyInfo);
                // Check if key is expired (if expiry info exists)
                if (info.expires && new Date(info.expires) < new Date()) {
                    console.log('Stored API key has expired');
                    this.clearApiKey();
                    return;
                }
                this.apiKey = stored;
            }
        } catch (error) {
            console.error('Error loading stored API key:', error);
            this.clearApiKey();
        }
    }

    /**
     * Store API key with metadata
     */
    storeApiKey(apiKey, metadata = {}) {
        try {
            this.apiKey = apiKey;
            localStorage.setItem('apiKey', apiKey);
            
            const keyInfo = {
                created: new Date().toISOString(),
                lastUsed: new Date().toISOString(),
                ...metadata
            };
            localStorage.setItem('apiKeyInfo', JSON.stringify(keyInfo));
            
            console.log('API key stored successfully');
        } catch (error) {
            console.error('Error storing API key:', error);
        }
    }

    /**
     * Clear stored API key
     */
    clearApiKey() {
        this.apiKey = null;
        localStorage.removeItem('apiKey');
        localStorage.removeItem('apiKeyInfo');
    }

    /**
     * Get current API key
     */
    getApiKey() {
        return this.apiKey;
    }

    /**
     * Check if we have a valid API key
     */
    hasValidApiKey() {
        return !!this.apiKey;
    }

    /**
     * Attempt to refresh/generate a new API key
     */
    async refreshApiKey() {
        if (this.isRefreshing) {
            // Return existing refresh promise
            return new Promise((resolve, reject) => {
                this.pendingRequests.push({ resolve, reject });
            });
        }

        this.isRefreshing = true;
        
        try {
            console.log('Attempting to refresh API key...');
            
            const response = await fetch(`${API_CONFIG?.baseURL || '/api'}/auth/register_api_key`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    name: `chat-session-${Date.now()}`, 
                    key_type: 'basic'  // Changed from 'chat_user' to 'basic'
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to refresh API key: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            const newApiKey = data.api_key;
            
            // Store the new key
            this.storeApiKey(newApiKey, {
                type: 'basic',
                source: 'auto_refresh'
            });

            // Resolve pending requests
            this.pendingRequests.forEach(({ resolve }) => resolve(newApiKey));
            this.pendingRequests = [];
            
            console.log('API key refreshed successfully');
            return newApiKey;
            
        } catch (error) {
            console.error('Error refreshing API key:', error);
            
            // Reject pending requests
            this.pendingRequests.forEach(({ reject }) => reject(error));
            this.pendingRequests = [];
            
            throw error;
        } finally {
            this.isRefreshing = false;
        }
    }

    /**
     * Make an authenticated API request with automatic retry and key refresh
     */
    async makeAuthenticatedRequest(url, options = {}, retryCount = 0) {
        // For free tier users, allow requests without API key
        // The backend should handle free tier limits
        if (!this.hasValidApiKey()) {
            // Don't try to refresh key for free users
            // Just make the request without authentication
            console.log('No API key available, proceeding as free tier user');
            
            const headers = {
                'Content-Type': 'application/json',
                ...options.headers
            };
            
            const requestOptions = {
                ...options,
                headers
            };
            
            return fetch(url, requestOptions);
        }

        // Prepare headers with authentication
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers,
            'Authorization': `Bearer ${this.apiKey}`
        };

        const requestOptions = {
            ...options,
            headers
        };

        try {
            const response = await fetch(url, requestOptions);
            
            // Handle 401/403 errors (authentication issues)
            if (response.status === 401 || response.status === 403) {
                // For free tier users, don't try to refresh API key
                // Just return the response and let the app handle it
                console.log('Authentication error for request, but proceeding as free tier user');
                return response;
            }

            // Update last used timestamp
            try {
                const keyInfo = JSON.parse(localStorage.getItem('apiKeyInfo') || '{}');
                keyInfo.lastUsed = new Date().toISOString();
                localStorage.setItem('apiKeyInfo', JSON.stringify(keyInfo));
            } catch (e) {
                // Ignore metadata update errors
            }

            return response;
            
        } catch (error) {
            if (error instanceof AuthenticationError) {
                throw error;
            }
            
            // Handle network errors with retry
            if (retryCount < this.maxRetries && this.isNetworkError(error)) {
                console.log(`Network error, retrying... (${retryCount + 1}/${this.maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, this.retryDelay * (retryCount + 1)));
                return this.makeAuthenticatedRequest(url, options, retryCount + 1);
            }
            
            throw error;
        }
    }

    /**
     * Check if error is a network-related error
     */
    isNetworkError(error) {
        return error instanceof TypeError && error.message.includes('fetch');
    }

    /**
     * Get authentication status for UI display
     */
    getAuthStatus() {
        if (!this.hasValidApiKey()) {
            return { 
                status: 'unauthenticated', 
                message: 'No API key available' 
            };
        }

        try {
            const keyInfo = JSON.parse(localStorage.getItem('apiKeyInfo') || '{}');
            return {
                status: 'authenticated',
                message: 'Authenticated',
                keyInfo: {
                    created: keyInfo.created,
                    lastUsed: keyInfo.lastUsed,
                    type: keyInfo.type || 'unknown'
                }
            };
        } catch (error) {
            return {
                status: 'authenticated',
                message: 'Authenticated (legacy key)'
            };
        }
    }
}

/**
 * Custom error class for authentication issues
 */
class AuthenticationError extends Error {
    constructor(message) {
        super(message);
        this.name = 'AuthenticationError';
    }
}

// Create global instance
const apiAuth = new APIAuthManager();

// Make it globally available
window.apiAuth = apiAuth;
window.AuthenticationError = AuthenticationError;
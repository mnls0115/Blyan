/**
 * Production-Grade API Authentication Manager V2
 * ==============================================
 * 
 * Complete rewrite with:
 * - Proper state management
 * - Automatic refresh with backoff
 * - Free tier support
 * - Comprehensive error handling
 * - Request queuing during refresh
 * - Analytics integration
 */

class APIAuthManagerV2 {
    constructor() {
        // Configuration
        this.config = {
            baseURL: window.API_CONFIG?.baseURL || '/api',
            refreshBeforeExpiry: 24 * 60 * 60 * 1000, // 24 hours in ms
            maxRetries: 3,
            retryDelays: [1000, 3000, 5000], // Progressive backoff
            requestTimeout: 30000, // 30 seconds
            
            // Storage keys
            storageKeys: {
                apiKey: 'blyan_api_key',
                keyInfo: 'blyan_api_key_info',
                userRole: 'blyan_user_role',
                refreshToken: 'blyan_refresh_token'
            }
        };

        // State management
        this.state = {
            apiKey: null,
            keyInfo: null,
            isRefreshing: false,
            refreshPromise: null,
            pendingRequests: [],
            requestStats: {
                total: 0,
                success: 0,
                failed: 0,
                rateLimited: 0
            }
        };

        // Initialize
        this.init();
    }

    /**
     * Initialize the auth manager
     */
    init() {
        this.loadStoredCredentials();
        this.setupPeriodicRefresh();
        this.setupVisibilityHandler();
        
        // Expose global error handler
        window.addEventListener('unhandledrejection', (event) => {
            if (event.reason?.name === 'AuthenticationError') {
                this.handleGlobalAuthError(event.reason);
            }
        });
    }

    /**
     * Load credentials from storage
     */
    loadStoredCredentials() {
        try {
            const storedKey = localStorage.getItem(this.config.storageKeys.apiKey);
            const storedInfo = localStorage.getItem(this.config.storageKeys.keyInfo);
            
            if (storedKey && storedInfo) {
                const keyInfo = JSON.parse(storedInfo);
                
                // Validate expiry
                if (keyInfo.expiresAt && new Date(keyInfo.expiresAt) > new Date()) {
                    this.state.apiKey = storedKey;
                    this.state.keyInfo = keyInfo;
                    console.log('[Auth] Loaded valid API key from storage');
                } else {
                    console.log('[Auth] Stored API key has expired');
                    this.clearCredentials();
                }
            }
        } catch (error) {
            console.error('[Auth] Error loading stored credentials:', error);
            this.clearCredentials();
        }
    }

    /**
     * Store credentials securely
     */
    storeCredentials(apiKey, keyInfo) {
        try {
            this.state.apiKey = apiKey;
            this.state.keyInfo = keyInfo;
            
            localStorage.setItem(this.config.storageKeys.apiKey, apiKey);
            localStorage.setItem(this.config.storageKeys.keyInfo, JSON.stringify(keyInfo));
            localStorage.setItem(this.config.storageKeys.userRole, keyInfo.role || 'basic');
            
            console.log('[Auth] Credentials stored successfully');
        } catch (error) {
            console.error('[Auth] Error storing credentials:', error);
        }
    }

    /**
     * Clear all stored credentials
     */
    clearCredentials() {
        this.state.apiKey = null;
        this.state.keyInfo = null;
        
        Object.values(this.config.storageKeys).forEach(key => {
            localStorage.removeItem(key);
        });
        
        console.log('[Auth] Credentials cleared');
    }

    /**
     * Check if user has valid API key
     */
    hasValidApiKey() {
        if (!this.state.apiKey || !this.state.keyInfo) {
            return false;
        }
        
        // Check expiry
        if (this.state.keyInfo.expiresAt) {
            const expiresAt = new Date(this.state.keyInfo.expiresAt);
            if (expiresAt <= new Date()) {
                console.log('[Auth] API key has expired');
                return false;
            }
        }
        
        return true;
    }

    /**
     * Get current user role
     */
    getUserRole() {
        return this.state.keyInfo?.role || 'free_tier';
    }

    /**
     * Check if user can access a specific scope
     */
    hasScope(requiredScope) {
        if (!this.state.keyInfo?.scopes) {
            return false;
        }
        return this.state.keyInfo.scopes.includes(requiredScope);
    }

    /**
     * Setup periodic refresh check
     */
    setupPeriodicRefresh() {
        // Check every hour
        setInterval(() => {
            if (this.shouldRefreshKey()) {
                this.refreshApiKey().catch(console.error);
            }
        }, 60 * 60 * 1000);
    }

    /**
     * Setup visibility change handler (refresh on focus)
     */
    setupVisibilityHandler() {
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.shouldRefreshKey()) {
                this.refreshApiKey().catch(console.error);
            }
        });
    }

    /**
     * Check if key should be refreshed
     */
    shouldRefreshKey() {
        if (!this.hasValidApiKey()) {
            return false;
        }
        
        const expiresAt = new Date(this.state.keyInfo.expiresAt);
        const refreshThreshold = new Date(Date.now() + this.config.refreshBeforeExpiry);
        
        return expiresAt <= refreshThreshold;
    }

    /**
     * Register a new API key
     */
    async registerApiKey(role = 'basic', metadata = {}) {
        console.log(`[Auth] Registering new ${role} API key`);
        
        const requestBody = {
            name: `web-client-${Date.now()}`,
            key_type: role,
            metadata: {
                ...metadata,
                userAgent: navigator.userAgent,
                timestamp: new Date().toISOString()
            }
        };

        try {
            const response = await fetch(`${this.config.baseURL}/auth/v2/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // Include current key if upgrading
                    ...(this.state.apiKey && {
                        'Authorization': `Bearer ${this.state.apiKey}`
                    })
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new AuthenticationError(
                    error.message || `Registration failed: ${response.status}`,
                    error.error || 'registration_failed',
                    error.details
                );
            }

            const data = await response.json();
            
            // Store the new credentials
            this.storeCredentials(data.api_key, {
                keyId: data.key_id,
                role: data.role,
                scopes: data.scopes,
                expiresAt: data.expires_at,
                refreshAfter: data.refresh_after,
                metadata: data.metadata
            });

            console.log(`[Auth] Successfully registered ${role} API key`);
            return data;
            
        } catch (error) {
            console.error('[Auth] Registration failed:', error);
            throw error;
        }
    }

    /**
     * Refresh existing API key
     */
    async refreshApiKey() {
        // Prevent concurrent refresh attempts
        if (this.state.isRefreshing) {
            return this.state.refreshPromise;
        }

        if (!this.hasValidApiKey()) {
            throw new AuthenticationError('No valid API key to refresh', 'no_key');
        }

        console.log('[Auth] Refreshing API key');
        this.state.isRefreshing = true;
        
        this.state.refreshPromise = this._performRefresh()
            .finally(() => {
                this.state.isRefreshing = false;
                this.state.refreshPromise = null;
            });

        return this.state.refreshPromise;
    }

    async _performRefresh() {
        try {
            const response = await fetch(`${this.config.baseURL}/auth/v2/refresh`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    current_key: this.state.apiKey
                })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                
                // If refresh fails, clear credentials
                if (response.status === 401) {
                    this.clearCredentials();
                }
                
                throw new AuthenticationError(
                    error.message || `Refresh failed: ${response.status}`,
                    error.error || 'refresh_failed',
                    error.details
                );
            }

            const data = await response.json();
            
            // Update credentials
            this.storeCredentials(data.api_key, {
                keyId: data.key_id,
                role: data.role,
                scopes: data.scopes,
                expiresAt: data.expires_at,
                refreshAfter: data.refresh_after,
                metadata: data.metadata
            });

            console.log('[Auth] API key refreshed successfully');
            
            // Process pending requests
            this.processPendingRequests();
            
            return data;
            
        } catch (error) {
            console.error('[Auth] Refresh failed:', error);
            
            // Notify pending requests of failure
            this.state.pendingRequests.forEach(({ reject }) => reject(error));
            this.state.pendingRequests = [];
            
            throw error;
        }
    }

    /**
     * Process queued requests after refresh
     */
    processPendingRequests() {
        const pending = this.state.pendingRequests;
        this.state.pendingRequests = [];
        
        pending.forEach(({ resolve }) => {
            resolve(this.state.apiKey);
        });
    }

    /**
     * Make an authenticated request with automatic retry and refresh
     */
    async makeAuthenticatedRequest(url, options = {}, retryCount = 0) {
        // Track request statistics
        this.state.requestStats.total++;
        
        // Determine if authentication is required
        const requiresAuth = this.shouldUseAuthentication(url, options);
        
        if (requiresAuth && !this.hasValidApiKey()) {
            // For free tier, proceed without auth
            if (this.getUserRole() === 'free_tier') {
                console.log('[Auth] Free tier request - no authentication');
                return this._performRequest(url, options, null);
            }
            
            // Try to register a basic key
            try {
                await this.registerApiKey('basic');
            } catch (error) {
                console.log('[Auth] Could not obtain API key, proceeding as free tier');
                return this._performRequest(url, options, null);
            }
        }

        // Check if refresh is needed
        if (requiresAuth && this.shouldRefreshKey()) {
            try {
                await this.refreshApiKey();
            } catch (error) {
                console.warn('[Auth] Refresh failed, using existing key');
            }
        }

        // Perform the request
        try {
            const response = await this._performRequest(
                url,
                options,
                requiresAuth ? this.state.apiKey : null
            );
            
            // Handle authentication errors
            if (response.status === 401 && requiresAuth && retryCount < this.config.maxRetries) {
                console.log('[Auth] Got 401, attempting refresh and retry');
                
                try {
                    await this.refreshApiKey();
                    return this.makeAuthenticatedRequest(url, options, retryCount + 1);
                } catch (refreshError) {
                    // If refresh fails, try without auth (free tier fallback)
                    console.log('[Auth] Refresh failed, falling back to free tier');
                    this.clearCredentials();
                    return this._performRequest(url, options, null);
                }
            }
            
            // Handle rate limiting
            if (response.status === 429) {
                this.state.requestStats.rateLimited++;
                const retryAfter = response.headers.get('X-RateLimit-Reset') || 60;
                
                throw new RateLimitError(
                    `Rate limit exceeded. Retry after ${retryAfter} seconds`,
                    parseInt(retryAfter)
                );
            }
            
            // Track success
            if (response.ok) {
                this.state.requestStats.success++;
            } else {
                this.state.requestStats.failed++;
            }
            
            return response;
            
        } catch (error) {
            this.state.requestStats.failed++;
            
            // Network error retry
            if (this.isNetworkError(error) && retryCount < this.config.maxRetries) {
                const delay = this.config.retryDelays[retryCount] || 5000;
                console.log(`[Auth] Network error, retrying in ${delay}ms`);
                
                await new Promise(resolve => setTimeout(resolve, delay));
                return this.makeAuthenticatedRequest(url, options, retryCount + 1);
            }
            
            throw error;
        }
    }

    /**
     * Perform the actual HTTP request
     */
    async _performRequest(url, options, apiKey) {
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };
        
        // Add authentication if provided
        if (apiKey) {
            headers['Authorization'] = `Bearer ${apiKey}`;
        }
        
        // Add request ID for tracking
        headers['X-Request-ID'] = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.config.requestTimeout);
        
        try {
            const response = await fetch(url, {
                ...options,
                headers,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            return response;
            
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new NetworkError('Request timeout', 'timeout');
            }
            
            throw error;
        }
    }

    /**
     * Determine if a request should use authentication
     */
    shouldUseAuthentication(url, options) {
        // Public endpoints that don't require auth
        const publicEndpoints = [
            '/health',
            '/status',
            '/auth/v2/register',
            '/public'
        ];
        
        // Check if endpoint is public
        const isPublic = publicEndpoints.some(endpoint => url.includes(endpoint));
        if (isPublic) {
            return false;
        }
        
        // Check if explicitly marked as no-auth
        if (options.noAuth === true) {
            return false;
        }
        
        // All other requests should use auth if available
        return true;
    }

    /**
     * Check if error is network-related
     */
    isNetworkError(error) {
        return error instanceof TypeError || 
               error instanceof NetworkError ||
               error.message?.includes('network') ||
               error.message?.includes('fetch');
    }

    /**
     * Handle global authentication errors
     */
    handleGlobalAuthError(error) {
        console.error('[Auth] Global authentication error:', error);
        
        // Show user-friendly notification
        this.showAuthNotification(error.message);
        
        // Emit custom event for app to handle
        window.dispatchEvent(new CustomEvent('auth:error', {
            detail: { error }
        }));
    }

    /**
     * Show authentication notification to user
     */
    showAuthNotification(message) {
        // Create notification element if it doesn't exist
        let notification = document.getElementById('auth-notification');
        
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'auth-notification';
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: linear-gradient(135deg, #ef4444, #dc2626);
                color: white;
                padding: 16px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                z-index: 10000;
                max-width: 400px;
                animation: slideIn 0.3s ease-out;
            `;
            document.body.appendChild(notification);
        }
        
        notification.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <div style="font-weight: 600; margin-bottom: 4px;">Authentication Issue</div>
                    <div style="font-size: 14px;">${message}</div>
                </div>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: none; border: none; color: white; cursor: pointer; font-size: 20px;">
                    ×
                </button>
            </div>
        `;
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    /**
     * Get current authentication status
     */
    getAuthStatus() {
        const hasKey = this.hasValidApiKey();
        const role = this.getUserRole();
        
        return {
            isAuthenticated: hasKey,
            role: role,
            scopes: this.state.keyInfo?.scopes || [],
            expiresAt: this.state.keyInfo?.expiresAt,
            canRefresh: this.shouldRefreshKey(),
            statistics: this.state.requestStats
        };
    }

    /**
     * Manual login with API key
     */
    async loginWithApiKey(apiKey) {
        // Validate the key format
        if (!apiKey || typeof apiKey !== 'string') {
            throw new AuthenticationError('Invalid API key format', 'invalid_format');
        }
        
        // Store temporarily
        this.state.apiKey = apiKey;
        
        // Validate with backend
        try {
            const response = await this._performRequest(
                `${this.config.baseURL}/auth/v2/validate`,
                { method: 'GET' },
                apiKey
            );
            
            if (!response.ok) {
                throw new AuthenticationError('Invalid API key', 'invalid_key');
            }
            
            const keyInfo = await response.json();
            this.storeCredentials(apiKey, keyInfo);
            
            console.log('[Auth] Successfully logged in with API key');
            return keyInfo;
            
        } catch (error) {
            this.clearCredentials();
            throw error;
        }
    }

    /**
     * Logout and clear credentials
     */
    logout() {
        this.clearCredentials();
        
        // Emit logout event
        window.dispatchEvent(new CustomEvent('auth:logout'));
        
        console.log('[Auth] User logged out');
    }
}

/**
 * Custom error classes
 */
class AuthenticationError extends Error {
    constructor(message, code = 'auth_error', details = null) {
        super(message);
        this.name = 'AuthenticationError';
        this.code = code;
        this.details = details;
    }
}

class RateLimitError extends Error {
    constructor(message, retryAfter) {
        super(message);
        this.name = 'RateLimitError';
        this.retryAfter = retryAfter;
    }
}

class NetworkError extends Error {
    constructor(message, code = 'network_error') {
        super(message);
        this.name = 'NetworkError';
        this.code = code;
    }
}

/**
 * Initialize global auth manager
 */
const apiAuth = new APIAuthManagerV2();

// Export for use in other modules (using standard names)
window.apiAuth = apiAuth;
window.AuthenticationError = AuthenticationError;
window.RateLimitError = RateLimitError;
window.NetworkError = NetworkError;

/**
 * Convenience functions for common use cases
 */
window.makeAuthenticatedRequest = (url, options) => {
    return apiAuth.makeAuthenticatedRequest(url, options);
};

window.getAuthStatus = () => {
    return apiAuth.getAuthStatus();
};

window.requireAuth = () => {
    if (!apiAuth.hasValidApiKey()) {
        throw new AuthenticationError('Authentication required', 'auth_required');
    }
};

window.requireScope = (scope) => {
    if (!apiAuth.hasScope(scope)) {
        throw new AuthenticationError(
            `Missing required scope: ${scope}`,
            'insufficient_scope',
            { required: scope, available: apiAuth.state.keyInfo?.scopes || [] }
        );
    }
};

console.log('[Auth] API Authentication Manager V2 initialized');
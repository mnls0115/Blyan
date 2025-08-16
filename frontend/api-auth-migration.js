/**
 * API Authentication Migration Script
 * ====================================
 * 
 * This script handles the migration from V1 to V2 API auth system
 * with automatic fallback support.
 */

class APIAuthMigration {
    constructor() {
        this.v2Available = false;
        this.authManager = null;
        this.checkV2Availability();
    }

    /**
     * Check if V2 API endpoints are available
     */
    async checkV2Availability() {
        try {
            const response = await fetch('/api/auth/v2/validate', {
                method: 'GET',
                headers: {
                    'Authorization': 'Bearer test'
                }
            });
            
            // If we get any response (even 401), V2 is available
            this.v2Available = response.status !== 404;
            
            if (this.v2Available) {
                console.log('[Migration] V2 API Key System detected - using enhanced authentication');
                this.loadV2System();
            } else {
                console.log('[Migration] V2 not available - using V1 authentication');
                this.useV1System();
            }
        } catch (error) {
            console.log('[Migration] V2 check failed - using V1 authentication');
            this.useV1System();
        }
    }

    /**
     * Load V2 authentication system
     */
    async loadV2System() {
        // Check if V2 script is already loaded
        if (window.apiAuthV2) {
            this.authManager = window.apiAuthV2;
            this.setupV2Compatibility();
            return;
        }

        // Dynamically load V2 script
        const script = document.createElement('script');
        script.src = '/frontend/api-auth-v2.js';
        script.onload = () => {
            this.authManager = window.apiAuthV2;
            this.setupV2Compatibility();
            console.log('[Migration] V2 authentication system loaded');
        };
        script.onerror = () => {
            console.error('[Migration] Failed to load V2 system, falling back to V1');
            this.useV1System();
        };
        document.head.appendChild(script);
    }

    /**
     * Use V1 authentication system
     */
    useV1System() {
        // V1 is already loaded as apiAuth
        this.authManager = window.apiAuth;
        console.log('[Migration] Using V1 authentication system');
    }

    /**
     * Setup V2 compatibility layer
     */
    setupV2Compatibility() {
        // Override global apiAuth with V2 if available
        if (this.v2Available && window.apiAuthV2) {
            // Create compatibility wrapper
            const v2Wrapper = {
                // Map V1 methods to V2
                hasValidApiKey: () => window.apiAuthV2.hasValidApiKey(),
                getApiKey: () => window.apiAuthV2.state.apiKey,
                storeApiKey: (key, metadata) => window.apiAuthV2.storeCredentials(key, metadata),
                clearApiKey: () => window.apiAuthV2.clearCredentials(),
                refreshApiKey: () => window.apiAuthV2.refreshApiKey(),
                makeAuthenticatedRequest: (url, options) => window.apiAuthV2.makeAuthenticatedRequest(url, options),
                getAuthStatus: () => window.apiAuthV2.getAuthStatus(),
                
                // V2-specific features
                loginWithApiKey: (key) => window.apiAuthV2.loginWithApiKey(key),
                logout: () => window.apiAuthV2.logout(),
                hasScope: (scope) => window.apiAuthV2.hasScope(scope),
                getUserRole: () => window.apiAuthV2.getUserRole()
            };

            // Replace global apiAuth with V2 wrapper
            window.apiAuth = v2Wrapper;
            
            // Migrate existing V1 keys to V2 format
            this.migrateV1Keys();
        }
    }

    /**
     * Migrate V1 API keys to V2 format
     */
    async migrateV1Keys() {
        const v1Key = localStorage.getItem('apiKey');
        const v1KeyInfo = localStorage.getItem('apiKeyInfo');
        
        if (v1Key && !localStorage.getItem('blyan_api_key')) {
            console.log('[Migration] Migrating V1 API key to V2 format');
            
            try {
                // Validate V1 key with V2 endpoint
                const response = await fetch('/api/auth/v2/validate', {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${v1Key}`
                    }
                });
                
                if (response.ok) {
                    const keyInfo = await response.json();
                    
                    // Store in V2 format
                    window.apiAuthV2.storeCredentials(v1Key, {
                        keyId: keyInfo.key_id,
                        role: keyInfo.role,
                        scopes: keyInfo.scopes,
                        expiresAt: keyInfo.expires_at,
                        metadata: keyInfo.metadata
                    });
                    
                    console.log('[Migration] V1 key successfully migrated to V2');
                    
                    // Clean up V1 storage
                    localStorage.removeItem('apiKey');
                    localStorage.removeItem('apiKeyInfo');
                } else {
                    console.log('[Migration] V1 key is invalid, will generate new V2 key on next request');
                }
            } catch (error) {
                console.error('[Migration] Error migrating V1 key:', error);
            }
        }
    }

    /**
     * Get the active auth manager
     */
    getAuthManager() {
        return this.authManager || window.apiAuth;
    }

    /**
     * Check if using V2 system
     */
    isV2Active() {
        return this.v2Available && window.apiAuthV2;
    }
}

// Initialize migration on load
const authMigration = new APIAuthMigration();

// Export for global use
window.authMigration = authMigration;

console.log('[Migration] API Authentication Migration Script loaded');
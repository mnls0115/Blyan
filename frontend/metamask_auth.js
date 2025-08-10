/**
 * MetaMask Authentication for Blyan Network
 * 
 * ### PHASE 1: MetaMask Only ###
 * Simple integration - 10 lines of code
 * 
 * ### PHASE 2: Dual Auth (coming) ###
 * MetaMask + Email OTP for non-crypto users
 * 
 * ### PHASE 3: Native BLY Wallet (future) ###
 * When DAU > 10,000
 */

class BlyanAuth {
    constructor() {
        this.apiUrl = window.location.hostname === 'localhost' 
            ? 'http://localhost:8000' 
            : `${window.location.protocol}//${window.location.hostname}/api`;
        
        this.account = null;
        this.token = null;
        this.userInfo = null;
    }

    /**
     * Check if MetaMask is installed
     */
    isMetaMaskInstalled() {
        return typeof window.ethereum !== 'undefined' && window.ethereum.isMetaMask;
    }

    /**
     * Connect to MetaMask and authenticate
     * 
     * @returns {Promise<Object>} User info with BLY balance
     */
    async connect() {
        // Check MetaMask
        if (!this.isMetaMaskInstalled()) {
            throw new Error('Please install MetaMask from https://metamask.io');
        }

        try {
            // 1. Request account access
            const accounts = await window.ethereum.request({ 
                method: 'eth_requestAccounts' 
            });
            
            if (!accounts || accounts.length === 0) {
                throw new Error('No accounts found');
            }

            this.account = accounts[0];
            console.log('Connected to account:', this.account);

            // 2. Request nonce from server
            const nonceResponse = await fetch(`${this.apiUrl}/wallet/request_nonce`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ address: this.account })
            });

            if (!nonceResponse.ok) {
                throw new Error('Failed to get nonce');
            }

            const { nonce, message } = await nonceResponse.json();

            // 3. Sign message with MetaMask
            const signature = await window.ethereum.request({
                method: 'personal_sign',
                params: [message, this.account]
            });

            console.log('Message signed successfully');

            // 4. Authenticate with server
            const authResponse = await fetch(`${this.apiUrl}/wallet/authenticate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    address: this.account,
                    signature: signature,
                    nonce: nonce,
                    message: message
                })
            });

            if (!authResponse.ok) {
                const error = await authResponse.json();
                throw new Error(error.detail || 'Authentication failed');
            }

            const authData = await authResponse.json();
            
            this.token = authData.token;
            this.userInfo = authData.user_info;

            // Store in localStorage for persistence
            localStorage.setItem('blyan_token', this.token);
            localStorage.setItem('blyan_account', this.account);

            console.log('✅ Authentication successful!');
            console.log('User info:', this.userInfo);

            return this.userInfo;

        } catch (error) {
            console.error('Authentication error:', error);
            
            // User rejected signature
            if (error.code === 4001) {
                throw new Error('Signature request rejected');
            }
            
            throw error;
        }
    }

    /**
     * Get BLY balance for current user
     */
    async getBalance() {
        if (!this.account) {
            throw new Error('Not connected');
        }

        const response = await fetch(`${this.apiUrl}/wallet/balance/${this.account}`);
        
        if (!response.ok) {
            throw new Error('Failed to get balance');
        }

        return await response.json();
    }

    /**
     * Logout and clear session
     */
    async logout() {
        if (this.account && this.token) {
            try {
                await fetch(`${this.apiUrl}/wallet/logout`, {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.token}`
                    },
                    body: JSON.stringify({ address: this.account })
                });
            } catch (error) {
                console.error('Logout error:', error);
            }
        }

        this.account = null;
        this.token = null;
        this.userInfo = null;
        
        localStorage.removeItem('blyan_token');
        localStorage.removeItem('blyan_account');
    }

    /**
     * Check if already authenticated
     */
    async checkAuth() {
        const token = localStorage.getItem('blyan_token');
        const account = localStorage.getItem('blyan_account');

        if (!token || !account) {
            return false;
        }

        try {
            const response = await fetch(`${this.apiUrl}/wallet/verify_token/${token}`);
            const data = await response.json();

            if (data.valid) {
                this.token = token;
                this.account = account;
                return true;
            }
        } catch (error) {
            console.error('Token verification error:', error);
        }

        return false;
    }

    /**
     * Listen for account changes
     */
    onAccountsChanged(callback) {
        if (window.ethereum) {
            window.ethereum.on('accountsChanged', (accounts) => {
                if (accounts.length === 0) {
                    // User disconnected wallet
                    this.logout();
                    callback(null);
                } else if (accounts[0] !== this.account) {
                    // User switched account
                    this.logout();
                    callback(accounts[0]);
                }
            });
        }
    }

    /**
     * Listen for chain changes
     */
    onChainChanged(callback) {
        if (window.ethereum) {
            window.ethereum.on('chainChanged', (chainId) => {
                callback(chainId);
                // Reload page on chain change (recommended by MetaMask)
                window.location.reload();
            });
        }
    }
}

// Export for use in other scripts
window.BlyanAuth = BlyanAuth;

/**
 * Usage Example:
 * 
 * const auth = new BlyanAuth();
 * 
 * // Connect button click
 * document.getElementById('connectBtn').onclick = async () => {
 *     try {
 *         const userInfo = await auth.connect();
 *         console.log('BLY Balance:', userInfo.bly_balance);
 *         console.log('Upload Credits:', userInfo.upload_credits);
 *     } catch (error) {
 *         alert(error.message);
 *     }
 * };
 * 
 * // Check if already logged in
 * if (await auth.checkAuth()) {
 *     console.log('Already authenticated');
 * }
 * 
 * // Listen for account changes
 * auth.onAccountsChanged((newAccount) => {
 *     if (newAccount) {
 *         console.log('Account changed to:', newAccount);
 *         // Re-authenticate with new account
 *     } else {
 *         console.log('Wallet disconnected');
 *     }
 * });
 */
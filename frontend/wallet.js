// Blyan Network Wallet Integration
// Connects user's blockchain wallet to track their BLY balance and contributions

class BlyanWallet {
    constructor() {
        this.address = null;
        this.balance = null;
        this.isConnected = false;
        this.contributorStats = null;
    }

    // Connect wallet (Web3 integration)
    async connect() {
        try {
            // Check if MetaMask or other Web3 wallet is available
            if (typeof window.ethereum !== 'undefined') {
                // Request account access
                const accounts = await window.ethereum.request({ 
                    method: 'eth_requestAccounts' 
                });
                
                this.address = accounts[0];
                this.isConnected = true;
                
                // Store in localStorage for persistence
                localStorage.setItem('blyan_wallet_address', this.address);
                
                // Fetch user stats
                await this.fetchUserStats();
                
                return {
                    success: true,
                    address: this.address
                };
            } else {
                return {
                    success: false,
                    error: 'No Web3 wallet detected. Please install MetaMask.'
                };
            }
        } catch (error) {
            console.error('Wallet connection error:', error);
            return {
                success: false,
                error: error.message
            };
        }
    }

    // Fetch user's contribution stats and balance
    async fetchUserStats() {
        if (!this.address) return;

        try {
            // In production, this would query the actual blockchain
            // For now, we'll use the API endpoint
            const response = await fetch(`/api/economy/user/${this.address}`);
            
            if (response.ok) {
                const data = await response.json();
                this.balance = data.balance;
                this.contributorStats = data.stats;
            } else {
                // User not found - new contributor
                this.balance = 0;
                this.contributorStats = {
                    rank: null,
                    total_earned: 0,
                    inference_earned: 0,
                    training_earned: 0,
                    first_contribution: null
                };
            }
        } catch (error) {
            console.error('Failed to fetch user stats:', error);
        }
    }

    // Get anonymized display for leaderboard
    getAnonymizedAddress() {
        if (!this.address) return null;
        return `${this.address.slice(0, 10)}...${this.address.slice(-8)}`;
    }

    // Check if current user is in the leaderboard
    async checkLeaderboardPosition() {
        if (!this.address) return null;

        try {
            const response = await fetch('/api/economy/leaderboard?limit=1000');
            const data = await response.json();
            
            // Find user's position (would be done server-side in production)
            const userEntry = data.entries.find(entry => 
                // In production, compare actual addresses or use privacy-preserving method
                entry.address_hash === this.getAnonymizedAddress()
            );
            
            return userEntry ? userEntry.rank : null;
        } catch (error) {
            console.error('Failed to check leaderboard:', error);
            return null;
        }
    }

    // Disconnect wallet
    disconnect() {
        this.address = null;
        this.balance = null;
        this.isConnected = false;
        this.contributorStats = null;
        localStorage.removeItem('blyan_wallet_address');
    }

    // Auto-reconnect on page load
    async autoConnect() {
        const savedAddress = localStorage.getItem('blyan_wallet_address');
        if (savedAddress && typeof window.ethereum !== 'undefined') {
            const accounts = await window.ethereum.request({ 
                method: 'eth_accounts' 
            });
            
            if (accounts.includes(savedAddress)) {
                this.address = savedAddress;
                this.isConnected = true;
                await this.fetchUserStats();
                return true;
            }
        }
        return false;
    }

    // Format BLY balance for display
    formatBalance(amount) {
        if (amount >= 1_000_000) {
            return `${(amount / 1_000_000).toFixed(2)}M BLY`;
        } else if (amount >= 1_000) {
            return `${(amount / 1_000).toFixed(2)}K BLY`;
        } else {
            return `${amount.toFixed(2)} BLY`;
        }
    }

    // Update UI with wallet info
    updateUI() {
        const walletButton = document.getElementById('wallet-connect-btn');
        const walletInfo = document.getElementById('wallet-info');
        const balanceDisplay = document.getElementById('wallet-balance');
        const addressDisplay = document.getElementById('wallet-address');
        
        if (this.isConnected) {
            // Update button
            if (walletButton) {
                walletButton.textContent = 'Disconnect';
                walletButton.classList.add('connected');
            }
            
            // Show wallet info
            if (walletInfo) {
                walletInfo.style.display = 'block';
            }
            
            // Display balance
            if (balanceDisplay) {
                balanceDisplay.textContent = this.formatBalance(this.balance || 0);
            }
            
            // Display address
            if (addressDisplay) {
                addressDisplay.textContent = this.getAnonymizedAddress();
            }
            
            // Update contributor badge if exists
            this.updateContributorBadge();
            
        } else {
            // Reset UI to disconnected state
            if (walletButton) {
                walletButton.textContent = 'Connect Wallet';
                walletButton.classList.remove('connected');
            }
            
            if (walletInfo) {
                walletInfo.style.display = 'none';
            }
        }
    }

    // Show contributor badge based on rank
    updateContributorBadge() {
        const badgeElement = document.getElementById('contributor-badge');
        if (!badgeElement || !this.contributorStats) return;
        
        const rank = this.contributorStats.rank;
        let badge = '';
        
        if (rank && rank <= 10) {
            badge = '🏆 Top 10 Contributor';
        } else if (rank && rank <= 100) {
            badge = '🥇 Top 100 Contributor';
        } else if (rank && rank <= 1000) {
            badge = '🥈 Active Contributor';
        } else if (this.contributorStats.total_earned > 0) {
            badge = '🥉 Contributor';
        }
        
        badgeElement.textContent = badge;
        badgeElement.style.display = badge ? 'inline-block' : 'none';
    }
}

// Initialize wallet on page load
const blyanWallet = new BlyanWallet();

document.addEventListener('DOMContentLoaded', async () => {
    // Auto-connect if previously connected
    await blyanWallet.autoConnect();
    blyanWallet.updateUI();
    
    // Setup connect button
    const connectBtn = document.getElementById('wallet-connect-btn');
    if (connectBtn) {
        connectBtn.addEventListener('click', async () => {
            if (blyanWallet.isConnected) {
                blyanWallet.disconnect();
            } else {
                const result = await blyanWallet.connect();
                if (!result.success) {
                    alert(result.error);
                }
            }
            blyanWallet.updateUI();
        });
    }
    
    // Listen for account changes
    if (window.ethereum) {
        window.ethereum.on('accountsChanged', (accounts) => {
            if (accounts.length === 0) {
                blyanWallet.disconnect();
            } else if (accounts[0] !== blyanWallet.address) {
                blyanWallet.connect();
            }
            blyanWallet.updateUI();
        });
    }
});

// Export for use in other modules
window.BlyanWallet = blyanWallet;
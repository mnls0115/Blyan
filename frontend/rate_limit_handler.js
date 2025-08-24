/**
 * Rate Limit Handler for BLYAN Frontend
 * Handles 429 responses and displays user-friendly messages
 */

class RateLimitHandler {
    constructor() {
        this.rateLimitInfo = null;
        this.retryTimer = null;
    }

    /**
     * Handle 429 response from API
     */
    handle429Response(response) {
        try {
            const data = response.data || response;
            
            this.rateLimitInfo = {
                tier: data.tier || 'free',
                limit: data.limit || 20,
                windowHours: data.window_hours || 5,
                retryAfter: data.retry_after || 3600,
                retryAt: data.retry_at || (Date.now() / 1000 + 3600),
                message: data.message || 'Rate limit exceeded'
            };
            
            this.showRateLimitMessage();
            this.startRetryTimer();
            
            return this.rateLimitInfo;
        } catch (error) {
            console.error('Error parsing rate limit response:', error);
            return {
                message: 'Too many requests. Please try again later.'
            };
        }
    }

    /**
     * Display rate limit message to user
     */
    showRateLimitMessage() {
        // Remove any existing message
        const existingAlert = document.getElementById('rate-limit-alert');
        if (existingAlert) {
            existingAlert.remove();
        }

        // Calculate retry time
        const retryTime = new Date(this.rateLimitInfo.retryAt * 1000);
        const now = new Date();
        const minutesLeft = Math.ceil((retryTime - now) / 60000);
        const hoursLeft = Math.floor(minutesLeft / 60);
        const minsLeft = minutesLeft % 60;

        // Create alert element
        const alert = document.createElement('div');
        alert.id = 'rate-limit-alert';
        alert.className = 'rate-limit-alert';
        alert.innerHTML = `
            <div class="alert-content">
                <h3>⏱️ Rate Limit Reached</h3>
                <p>You've used all ${this.rateLimitInfo.limit} free requests in the ${this.rateLimitInfo.windowHours}-hour window.</p>
                <p class="retry-time">Try again in: <span id="retry-countdown">${hoursLeft}h ${minsLeft}m</span></p>
                <div class="alert-actions">
                    <button onclick="rateLimitHandler.checkStatus()" class="btn-secondary">Check Status</button>
                    <button onclick="rateLimitHandler.dismiss()" class="btn-primary">OK</button>
                </div>
                <p class="upgrade-hint">💡 Tip: Connect your wallet for premium tier (1000 requests/hour)</p>
            </div>
        `;

        // Add styles if not already present
        if (!document.getElementById('rate-limit-styles')) {
            const style = document.createElement('style');
            style.id = 'rate-limit-styles';
            style.textContent = `
                .rate-limit-alert {
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px;
                    padding: 2px;
                    z-index: 10000;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    animation: slideIn 0.3s ease-out;
                }
                
                .alert-content {
                    background: #1a1a2e;
                    border-radius: 10px;
                    padding: 30px;
                    max-width: 400px;
                    text-align: center;
                    color: white;
                }
                
                .alert-content h3 {
                    margin: 0 0 15px 0;
                    font-size: 24px;
                    color: #fff;
                }
                
                .alert-content p {
                    margin: 10px 0;
                    color: #ccc;
                    line-height: 1.6;
                }
                
                .retry-time {
                    font-size: 18px;
                    color: #ffd700;
                    margin: 20px 0;
                }
                
                #retry-countdown {
                    font-weight: bold;
                    font-size: 20px;
                }
                
                .alert-actions {
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                    margin: 20px 0;
                }
                
                .alert-actions button {
                    padding: 10px 20px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 14px;
                    transition: all 0.3s;
                }
                
                .btn-primary {
                    background: #667eea;
                    color: white;
                }
                
                .btn-primary:hover {
                    background: #5a6fd8;
                }
                
                .btn-secondary {
                    background: transparent;
                    color: #667eea;
                    border: 1px solid #667eea;
                }
                
                .btn-secondary:hover {
                    background: rgba(102, 126, 234, 0.1);
                }
                
                .upgrade-hint {
                    font-size: 12px;
                    color: #aaa;
                    margin-top: 15px;
                    padding-top: 15px;
                    border-top: 1px solid #333;
                }
                
                @keyframes slideIn {
                    from {
                        opacity: 0;
                        transform: translate(-50%, -45%);
                    }
                    to {
                        opacity: 1;
                        transform: translate(-50%, -50%);
                    }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(alert);
    }

    /**
     * Start countdown timer
     */
    startRetryTimer() {
        if (this.retryTimer) {
            clearInterval(this.retryTimer);
        }

        this.retryTimer = setInterval(() => {
            const countdown = document.getElementById('retry-countdown');
            if (!countdown) {
                clearInterval(this.retryTimer);
                return;
            }

            const retryTime = new Date(this.rateLimitInfo.retryAt * 1000);
            const now = new Date();
            const secondsLeft = Math.max(0, Math.floor((retryTime - now) / 1000));

            if (secondsLeft <= 0) {
                countdown.textContent = 'Ready!';
                clearInterval(this.retryTimer);
                setTimeout(() => this.dismiss(), 2000);
            } else {
                const hours = Math.floor(secondsLeft / 3600);
                const minutes = Math.floor((secondsLeft % 3600) / 60);
                const seconds = secondsLeft % 60;
                
                if (hours > 0) {
                    countdown.textContent = `${hours}h ${minutes}m`;
                } else if (minutes > 0) {
                    countdown.textContent = `${minutes}m ${seconds}s`;
                } else {
                    countdown.textContent = `${seconds}s`;
                }
            }
        }, 1000);
    }

    /**
     * Check current rate limit status
     */
    async checkStatus() {
        try {
            const response = await fetch(`${API_CONFIG.baseURL}/rate-limit/status`);
            const data = await response.json();
            
            alert(`Rate Limit Status:\n\nTier: ${data.rate_limit.tier}\nRemaining: ${data.rate_limit.remaining}/${data.rate_limit.limit}\nWindow: ${data.rate_limit.window_hours} hours\n\n${data.message}`);
        } catch (error) {
            console.error('Failed to check rate limit status:', error);
            alert('Failed to check rate limit status. Please try again.');
        }
    }

    /**
     * Dismiss the rate limit alert
     */
    dismiss() {
        const alert = document.getElementById('rate-limit-alert');
        if (alert) {
            alert.style.animation = 'slideIn 0.3s ease-out reverse';
            setTimeout(() => alert.remove(), 300);
        }
        
        if (this.retryTimer) {
            clearInterval(this.retryTimer);
            this.retryTimer = null;
        }
    }

    /**
     * Check if currently rate limited
     */
    isRateLimited() {
        if (!this.rateLimitInfo) return false;
        
        const now = Date.now() / 1000;
        return now < this.rateLimitInfo.retryAt;
    }
}

// Create global instance
window.rateLimitHandler = new RateLimitHandler();

// Intercept fetch to handle 429 responses automatically
const originalFetch = window.fetch;
window.fetch = async function(...args) {
    const response = await originalFetch.apply(this, args);
    
    if (response.status === 429) {
        const clonedResponse = response.clone();
        try {
            const data = await clonedResponse.json();
            window.rateLimitHandler.handle429Response(data);
        } catch (e) {
            console.error('Failed to parse 429 response:', e);
        }
    }
    
    return response;
};
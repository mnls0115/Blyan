/**
 * Production-optimized chat interface
 * No mock data, streamlined flow, better error handling
 */

class OptimizedChat {
    constructor(apiUrl = '') {
        this.apiUrl = apiUrl || window.location.origin;
        // Use API_CONFIG if available
        if (typeof window !== 'undefined' && window.API_CONFIG) {
            this.apiUrl = window.API_CONFIG.baseURL;
        }
        this.isLoading = false;
        this.messageHistory = [];
        this.useProduction = true; // Use production endpoint by default
        this.activeTicket = null;
        this.sseController = null;
        this.queuePollInterval = null;
    }

    /**
     * Send message to API with optimized flow and queue support
     */
    async sendMessage(prompt, options = {}) {
        if (this.isLoading) {
            console.warn('Request already in progress');
            return null;
        }

        // Clamp prompt length
        const clampedPrompt = this._clampPrompt(prompt);
        if (clampedPrompt !== prompt) {
            return {
                prompt: prompt,
                response: 'Prompt exceeds maximum length of 8192 characters. Please shorten your message.',
                error: true,
                latency: 0
            };
        }

        this.isLoading = true;
        const startTime = Date.now();

        try {
            // Determine endpoint
            const endpoint = this.useProduction ? '/chat/production' : '/chat';
            
            // Build request
            const requestBody = {
                prompt: prompt,
                max_new_tokens: options.maxTokens || 100,
                stream: options.stream || false
            };

            // Send request
            const response = await fetch(`${this.apiUrl}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // Add user address if available
                    ...(this.userAddress && { 'X-User-Address': this.userAddress })
                },
                body: JSON.stringify(requestBody)
            });

            // Handle capacity (503) with queue
            if (response.status === 503) {
                const retryAfter = this._parseRetryAfter(response);
                if (retryAfter !== null) {
                    // Switch to queue flow (keep isLoading true for UI consistency)
                    return await this._handleQueueFlow(prompt, options);
                }
            }

            // Handle response
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || error.message || 'Request failed');
            }

            const data = await response.json();
            
            // Process successful response
            const result = {
                prompt: prompt,
                response: data.response || data.error || 'No response received',
                latency: Date.now() - startTime,
                requestId: data.request_id,
                cached: data.cache_hit || false,
                tokensGenerated: data.tokens_generated,
                expertUsage: data.expert_usage
            };

            // Store in history
            this.messageHistory.push(result);

            return result;

        } catch (error) {
            console.error('Chat request failed:', error);
            
            // User-friendly error handling
            const errorResult = {
                prompt: prompt,
                response: this._getErrorMessage(error),
                latency: Date.now() - startTime,
                error: true
            };

            this.messageHistory.push(errorResult);
            return errorResult;

        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Get user-friendly error message
     */
    _getErrorMessage(error) {
        // Network errors
        if (error.message.includes('fetch')) {
            return 'Unable to connect to the server. Please check your connection.';
        }
        
        // Rate limiting
        if (error.message.includes('429') || error.message.includes('rate')) {
            return 'Too many requests. Please wait a moment and try again.';
        }
        
        // Model unavailable (no fallback to inferior models)
        if (error.message.includes('not available') || error.message.includes('503')) {
            return 'Model inference is not available at this time. Please try again later.';
        }
        
        // Generic error
        return 'An error occurred. Please try again.';
    }

    /**
     * Stream response with optimized handling
     */
    async streamMessage(prompt, onChunk, options = {}) {
        if (this.isLoading) {
            console.warn('Request already in progress');
            return null;
        }

        this.isLoading = true;
        const startTime = Date.now();

        try {
            const response = await fetch(`${this.apiUrl}/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.userAddress && { 'X-User-Address': this.userAddress })
                },
                body: JSON.stringify({
                    prompt: prompt,
                    max_new_tokens: options.maxTokens || 100,
                    stream: true
                })
            });

            if (!response.ok) {
                throw new Error(`Stream failed: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                fullResponse += chunk;

                // Call callback with chunk
                if (onChunk) {
                    onChunk(chunk);
                }
            }

            return {
                prompt: prompt,
                response: fullResponse,
                latency: Date.now() - startTime,
                streamed: true
            };

        } catch (error) {
            console.error('Stream failed:', error);
            return {
                prompt: prompt,
                response: this._getErrorMessage(error),
                error: true
            };
        } finally {
            this.isLoading = false;
        }
    }

    /**
     * Get production metrics
     */
    async getMetrics() {
        try {
            const response = await fetch(`${this.apiUrl}/metrics/production`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Failed to fetch metrics:', error);
        }
        return null;
    }

    /**
     * Clear message history
     */
    clearHistory() {
        this.messageHistory = [];
    }

    /**
     * Set user address for authenticated requests
     */
    setUserAddress(address) {
        this.userAddress = address;
    }

    /**
     * Toggle between production and standard endpoint
     */
    toggleProductionMode(useProduction = true) {
        this.useProduction = useProduction;
    }

    /**
     * Handle queue flow when capacity is reached
     */
    async _handleQueueFlow(prompt, options = {}) {
        try {
            // Enqueue the request
            const queueResult = await this._enqueue(prompt, options);
            if (!queueResult || queueResult.error) {
                return queueResult;
            }

            // Store active ticket
            this.activeTicket = queueResult.ticketId;

            // Show queue UI
            this._showQueuePanel({
                position: queueResult.position,
                eta: queueResult.eta
            });

            // Start SSE monitoring
            return await this._startSSE(queueResult.ticketId, prompt, options);

        } catch (error) {
            console.error('Queue flow failed:', error);
            this._hideQueuePanel();
            this.isLoading = false; // Reset loading state on queue failure
            return {
                prompt: prompt,
                response: 'Failed to join queue. Please try again.',
                error: true
            };
        }
    }

    /**
     * Enqueue a chat request
     */
    async _enqueue(prompt, options, retryCount = 0) {
        const maxRetries = 3;
        
        try {
            const response = await fetch(`${this.apiUrl}/chat/admit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.userAddress && { 'X-User-Address': this.userAddress })
                },
                body: JSON.stringify({
                    prompt: prompt,
                    max_new_tokens: options.maxTokens || 100
                })
            });

            if (response.ok) {
                const data = await response.json();
                return {
                    ticketId: data.ticket_id,
                    position: data.position,
                    eta: data.eta_seconds,
                    state: data.state
                };
            }

            // Handle rate limiting and queue full
            if (response.status === 429 || response.status === 503) {
                const retryAfter = this._parseRetryAfter(response);
                const waitTime = Math.min(retryAfter || 10, 120) * 1000;
                
                if (retryCount < maxRetries) {
                    this._updateQueuePanel({
                        position: '--',
                        eta: Math.ceil(waitTime / 1000),
                        state: 'waiting'
                    }, `High load, retrying in ${Math.ceil(waitTime / 1000)}s...`);
                    
                    await this._sleep(waitTime);
                    return await this._enqueue(prompt, options, retryCount + 1);
                }
            }

            throw new Error(`Failed to enqueue: ${response.status}`);

        } catch (error) {
            console.error('Enqueue failed:', error);
            return {
                prompt: prompt,
                response: 'Failed to join queue. Please try again.',
                error: true
            };
        }
    }

    /**
     * Start SSE connection for queue updates
     */
    async _startSSE(ticketId, prompt, options) {
        return new Promise((resolve) => {
            try {
                // Close any existing SSE
                if (this.sseController) {
                    this.sseController.close();
                }

                const eventSource = new EventSource(`${this.apiUrl}/queue/stream?ticket_id=${ticketId}`);
                this.sseController = eventSource;

                eventSource.onmessage = async (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        
                        // Check for ready event
                        if (data.event === 'ready_to_start' || data.state === 'started') {
                            eventSource.close();
                            this.sseController = null;
                            
                            // Start the ticket
                            const result = await this._startTicket(ticketId, prompt, options);
                            this._hideQueuePanel();
                            this.isLoading = false; // Reset loading state when done
                            resolve(result);
                            return;
                        }

                        // Check for error states
                        if (data.state === 'failed' || data.state === 'expired' || data.state === 'cancelled' || data.state === 'canceled') {
                            eventSource.close();
                            this.sseController = null;
                            this._hideQueuePanel();
                            this.isLoading = false; // Reset loading state on error
                            
                            resolve({
                                prompt: prompt,
                                response: `Queue request ${data.state}. Please try again.`,
                                error: true
                            });
                            return;
                        }

                        // Update queue position
                        this._updateQueuePanel({
                            position: data.position,
                            eta: data.eta_seconds,
                            state: data.state
                        });

                    } catch (error) {
                        console.error('SSE message parse error:', error);
                    }
                };

                eventSource.onerror = () => {
                    console.warn('SSE connection failed, falling back to polling');
                    eventSource.close();
                    this.sseController = null;
                    
                    // Fall back to polling
                    this._pollQueueStatus(ticketId, prompt, options).then(resolve);
                };

            } catch (error) {
                console.error('SSE setup failed:', error);
                // Fall back to polling
                this._pollQueueStatus(ticketId, prompt, options).then(resolve);
            }
        });
    }

    /**
     * Poll queue status (fallback when SSE fails)
     */
    async _pollQueueStatus(ticketId, prompt, options) {
        const pollInterval = 1000; // 1 second
        
        return new Promise((resolve) => {
            this.queuePollInterval = setInterval(async () => {
                try {
                    const response = await fetch(`${this.apiUrl}/queue/status/${ticketId}`);
                    
                    if (!response.ok) {
                        clearInterval(this.queuePollInterval);
                        this._hideQueuePanel();
                        resolve({
                            prompt: prompt,
                            response: 'Queue status check failed. Please try again.',
                            error: true
                        });
                        return;
                    }

                    const data = await response.json();
                    
                    // Check if ready to start
                    if (data.state === 'started' || data.state === 'assigned') {
                        clearInterval(this.queuePollInterval);
                        const result = await this._startTicket(ticketId, prompt, options);
                        this._hideQueuePanel();
                        this.isLoading = false; // Reset loading state when done
                        resolve(result);
                        return;
                    }

                    // Check for error states
                    if (data.state === 'failed' || data.state === 'expired' || data.state === 'cancelled' || data.state === 'canceled') {
                        clearInterval(this.queuePollInterval);
                        this._hideQueuePanel();
                        this.isLoading = false; // Reset loading state on error
                        resolve({
                            prompt: prompt,
                            response: `Queue request ${data.state}. Please try again.`,
                            error: true
                        });
                        return;
                    }

                    // Update UI
                    this._updateQueuePanel({
                        position: data.position,
                        eta: data.eta_seconds,
                        state: data.state
                    });

                } catch (error) {
                    console.error('Queue poll error:', error);
                }
            }, pollInterval);
        });
    }

    /**
     * Start processing an admitted ticket
     */
    async _startTicket(ticketId, prompt, options, retryCount = 0) {
        const maxRetries = 5;
        const startTime = Date.now();
        
        try {
            const response = await fetch(`${this.apiUrl}/chat/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.userAddress && { 'X-User-Address': this.userAddress })
                },
                body: JSON.stringify({ ticket_id: ticketId })
            });

            if (response.ok) {
                const data = await response.json();
                this.activeTicket = null;
                
                return {
                    prompt: prompt,
                    response: data.response || 'No response received',
                    latency: Date.now() - startTime,
                    requestId: data.request_id,
                    cached: data.cache_hit || false,
                    tokensGenerated: data.tokens_generated
                };
            }

            // Handle capacity during start
            if (response.status === 503 && retryCount < maxRetries) {
                const retryAfter = this._parseRetryAfter(response);
                const waitTime = Math.min(retryAfter || 3, 30) * 1000;
                
                this._updateQueuePanel({
                    position: 0,
                    eta: Math.ceil(waitTime / 1000),
                    state: 'starting'
                }, `Starting soon, retrying in ${Math.ceil(waitTime / 1000)}s...`);
                
                await this._sleep(waitTime);
                return await this._startTicket(ticketId, prompt, options, retryCount + 1);
            }

            throw new Error(`Failed to start ticket: ${response.status}`);

        } catch (error) {
            console.error('Start ticket failed:', error);
            return {
                prompt: prompt,
                response: 'Failed to start processing. Please try again.',
                error: true
            };
        }
    }

    /**
     * Cancel a queued ticket
     */
    async _cancelTicket(ticketId) {
        if (!ticketId) return;
        
        try {
            await fetch(`${this.apiUrl}/queue/cancel/${ticketId}`, {
                method: 'POST'
            });
        } catch (error) {
            console.error('Cancel failed:', error);
        }
        
        this.activeTicket = null;
        this._hideQueuePanel();
        
        // Clean up SSE and polling
        if (this.sseController) {
            this.sseController.close();
            this.sseController = null;
        }
        if (this.queuePollInterval) {
            clearInterval(this.queuePollInterval);
            this.queuePollInterval = null;
        }
    }

    /**
     * Parse Retry-After header
     */
    _parseRetryAfter(response) {
        const retryAfter = response.headers.get('Retry-After');
        if (!retryAfter) return null;
        
        const seconds = parseInt(retryAfter, 10);
        return isNaN(seconds) ? null : seconds;
    }

    /**
     * Clamp prompt to maximum length
     */
    _clampPrompt(prompt, maxLen = 8192) {
        if (prompt.length <= maxLen) return prompt;
        return prompt.substring(0, maxLen);
    }

    /**
     * Sleep helper
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Show queue panel UI - Override these methods in your implementation
     */
    _showQueuePanel(info) {
        // This method should be overridden by the page using OptimizedChat
        console.log('Queue panel show:', info);
    }

    /**
     * Update queue panel UI - Override these methods in your implementation
     */
    _updateQueuePanel(info, customMessage = null) {
        // This method should be overridden by the page using OptimizedChat
        console.log('Queue panel update:', info, customMessage);
    }

    /**
     * Hide queue panel UI - Override these methods in your implementation
     */
    _hideQueuePanel() {
        // This method should be overridden by the page using OptimizedChat
        console.log('Queue panel hide');
        this.activeTicket = null;
    }

    /**
     * Get queue metrics
     */
    async getQueueMetrics() {
        try {
            const response = await fetch(`${this.apiUrl}/queue/metrics`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Failed to fetch queue metrics:', error);
        }
        return null;
    }
}

// Export for use in HTML
if (typeof window !== 'undefined') {
    window.OptimizedChat = OptimizedChat;
}

/**
 * Production-optimized chat interface
 * No mock data, streamlined flow, better error handling
 */

class OptimizedChat {
    constructor(apiUrl = '') {
        this.apiUrl = apiUrl || window.location.origin;
        this.isLoading = false;
        this.messageHistory = [];
        this.useProduction = true; // Use production endpoint by default
    }

    /**
     * Send message to API with optimized flow
     */
    async sendMessage(prompt, options = {}) {
        if (this.isLoading) {
            console.warn('Request already in progress');
            return null;
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
}

// Export for use in HTML
if (typeof window !== 'undefined') {
    window.OptimizedChat = OptimizedChat;
}
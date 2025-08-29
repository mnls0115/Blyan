#!/usr/bin/env node
/**
 * Blyan Dynamic API Proxy
 * 
 * Production-ready reverse proxy with:
 * - Dynamic node registry with persistence
 * - Health checks and automatic failover
 * - SSE/streaming support
 * - Metrics and monitoring
 * - Admin API with authentication
 */

const express = require('express');
const httpProxy = require('http-proxy');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
require('dotenv').config();

const app = express();
const proxy = httpProxy.createProxyServer({
    followRedirects: true,
    changeOrigin: true,
    secure: true,
    xfwd: true,
    ws: true  // WebSocket support
});

// Configuration
const CONFIG = {
    PORT: parseInt(process.env.PROXY_PORT || '9000'),
    HEALTH_INTERVAL_MS: parseInt(process.env.HEALTH_INTERVAL_MS || '5000'),
    HEALTH_TIMEOUT_MS: parseInt(process.env.HEALTH_TIMEOUT_MS || '3000'),
    HEALTH_PATHS: (process.env.HEALTH_PATHS || '/health,/pol/status').split(','),
    STATE_FILE: process.env.STATE_FILE || './data/proxy_state.json',
    ADMIN_TOKEN: process.env.ADMIN_TOKEN || crypto.randomBytes(32).toString('hex'),
    LOG_LEVEL: process.env.LOG_LEVEL || 'info',
    MAX_RETRIES: parseInt(process.env.MAX_RETRIES || '3'),
    RETRY_DELAY_MS: parseInt(process.env.RETRY_DELAY_MS || '1000'),
    ENABLE_METRICS: process.env.ENABLE_METRICS === 'true',
    SELECTION_POLICY: process.env.SELECTION_POLICY || 'active-first', // active-first, round-robin, least-latency
    PROXY_TIMEOUT: parseInt(process.env.PROXY_TIMEOUT || '600000'), // 10 minutes for long streams
    WS_HEARTBEAT_MS: parseInt(process.env.WS_HEARTBEAT_MS || '30000') // WebSocket keepalive
};

// Ensure admin token is set
if (!process.env.ADMIN_TOKEN) {
    console.warn('⚠️  No ADMIN_TOKEN set in environment. Generated:', CONFIG.ADMIN_TOKEN);
    console.warn('   Set ADMIN_TOKEN in .env for production');
}

// Track active WebSocket connections globally for cleanup
const activeWebSockets = new Map();

// State management
class NodeRegistry {
    constructor() {
        this.nodes = [];
        this.activeIndex = 0;
        this.roundRobinIndex = 0;
        this.metrics = {
            totalRequests: 0,
            failedRequests: 0,
            nodeStats: {}
        };
    }

    async load() {
        try {
            const stateFile = path.resolve(CONFIG.STATE_FILE);
            const data = await fs.readFile(stateFile, 'utf8');
            const state = JSON.parse(data);
            this.nodes = state.nodes || [];
            this.activeIndex = state.activeIndex || 0;
            console.log(`✅ Loaded ${this.nodes.length} nodes from state file`);
        } catch (err) {
            if (err.code !== 'ENOENT') {
                console.error('Failed to load state:', err);
            }
            // Initialize with environment nodes if available
            this.initFromEnvironment();
        }
    }

    initFromEnvironment() {
        // Load initial nodes from environment variables
        // Format: NODE_0=name|url, NODE_1=name|url, etc.
        const envNodes = [];
        for (let i = 0; i < 10; i++) {
            const nodeConfig = process.env[`NODE_${i}`];
            if (nodeConfig) {
                const [name, baseURL] = nodeConfig.split('|');
                if (name && baseURL) {
                    envNodes.push({
                        id: crypto.randomBytes(8).toString('hex'),
                        name,
                        baseURL: baseURL.replace(/\/+$/, ''),
                        healthy: false,
                        lastCheck: 0,
                        responseTime: null,
                        consecutiveFailures: 0,
                        metadata: {}
                    });
                }
            }
        }
        if (envNodes.length > 0) {
            this.nodes = envNodes;
            console.log(`✅ Initialized ${envNodes.length} nodes from environment`);
        }
    }

    async save() {
        try {
            const stateFile = path.resolve(CONFIG.STATE_FILE);
            await fs.mkdir(path.dirname(stateFile), { recursive: true });
            const state = {
                nodes: this.nodes,
                activeIndex: this.activeIndex,
                savedAt: new Date().toISOString()
            };
            await fs.writeFile(stateFile, JSON.stringify(state, null, 2));
        } catch (err) {
            console.error('Failed to save state:', err);
        }
    }

    addNode(name, baseURL, metadata = {}) {
        const node = {
            id: crypto.randomBytes(8).toString('hex'),
            name,
            baseURL: baseURL.replace(/\/+$/, ''),
            healthy: false,
            lastCheck: 0,
            responseTime: null,
            consecutiveFailures: 0,
            metadata,
            addedAt: new Date().toISOString()
        };
        this.nodes.push(node);
        this.save();
        return node;
    }

    removeNode(id) {
        const index = this.nodes.findIndex(n => n.id === id);
        if (index === -1) return false;
        
        this.nodes.splice(index, 1);
        if (this.activeIndex >= this.nodes.length) {
            this.activeIndex = 0;
        }
        this.save();
        return true;
    }

    updateNodeHealth(node, healthy, responseTime = null) {
        node.healthy = healthy;
        node.lastCheck = Date.now();
        node.responseTime = responseTime;
        
        if (!healthy) {
            node.consecutiveFailures++;
        } else {
            node.consecutiveFailures = 0;
        }

        // Track metrics
        if (!this.metrics.nodeStats[node.id]) {
            this.metrics.nodeStats[node.id] = {
                requests: 0,
                failures: 0,
                totalResponseTime: 0
            };
        }
    }

    selectNode(req) {
        // Check for override
        const url = new URL(`http://dummy${req.originalUrl}`);
        const targetOverride = url.searchParams.get('target');
        if (targetOverride && this.validateUrl(targetOverride)) {
            return targetOverride.replace(/\/+$/, '');
        }

        // Filter healthy nodes
        const healthyNodes = this.nodes.filter(n => n.healthy);
        if (healthyNodes.length === 0) {
            // If no healthy nodes, try the active one anyway
            return this.nodes[this.activeIndex]?.baseURL || null;
        }

        // Selection based on policy
        let selected;
        switch (CONFIG.SELECTION_POLICY) {
            case 'round-robin':
                selected = healthyNodes[this.roundRobinIndex % healthyNodes.length];
                this.roundRobinIndex++;
                break;
            
            case 'least-latency':
                selected = healthyNodes.reduce((best, node) => {
                    if (!best) return node;
                    if (node.responseTime === null) return best;
                    if (best.responseTime === null) return node;
                    return node.responseTime < best.responseTime ? node : best;
                });
                break;
            
            case 'active-first':
            default:
                // Use active node if healthy, otherwise first healthy
                const activeNode = this.nodes[this.activeIndex];
                if (activeNode && activeNode.healthy) {
                    selected = activeNode;
                } else {
                    selected = healthyNodes[0];
                }
                break;
        }

        return selected?.baseURL || null;
    }

    validateUrl(url) {
        try {
            const parsed = new URL(url);
            return ['http:', 'https:'].includes(parsed.protocol);
        } catch {
            return false;
        }
    }

    getStatus() {
        return {
            policy: CONFIG.SELECTION_POLICY,
            activeIndex: this.activeIndex,
            nodes: this.nodes.map(n => ({
                id: n.id,
                name: n.name,
                baseURL: n.baseURL,
                healthy: n.healthy,
                lastCheck: n.lastCheck,
                responseTime: n.responseTime,
                consecutiveFailures: n.consecutiveFailures,
                metadata: n.metadata
            })),
            metrics: this.metrics
        };
    }
}

// Initialize registry
const registry = new NodeRegistry();

// Health check implementation
class HealthChecker {
    constructor(registry) {
        this.registry = registry;
        this.checking = false;
    }

    async checkNode(node) {
        const startTime = Date.now();
        
        for (const healthPath of CONFIG.HEALTH_PATHS) {
            try {
                const url = `${node.baseURL}${healthPath}`;
                const response = await axios.get(url, {
                    timeout: CONFIG.HEALTH_TIMEOUT_MS,
                    validateStatus: () => true,
                    headers: {
                        'User-Agent': 'Blyan-Proxy-HealthCheck/1.0'
                    }
                });

                const responseTime = Date.now() - startTime;
                const healthy = response.status >= 200 && response.status < 500;
                
                this.registry.updateNodeHealth(node, healthy, responseTime);
                
                if (healthy) {
                    console.log(`✅ ${node.name} healthy (${responseTime}ms)`);
                    return;
                }
            } catch (err) {
                // Try next health path
                continue;
            }
        }
        
        // All health checks failed
        this.registry.updateNodeHealth(node, false);
        console.log(`❌ ${node.name} unhealthy`);
    }

    async checkAll() {
        if (this.checking) return;
        this.checking = true;

        try {
            await Promise.all(
                this.registry.nodes.map(node => this.checkNode(node))
            );
        } finally {
            this.checking = false;
        }
    }

    start() {
        // Initial check
        this.checkAll();
        
        // Periodic checks
        setInterval(() => this.checkAll(), CONFIG.HEALTH_INTERVAL_MS);
    }
}

// Proxy middleware
function proxyMiddleware(req, res) {
    const targetBase = registry.selectNode(req);
    
    if (!targetBase) {
        registry.metrics.failedRequests++;
        return res.status(503).json({
            error: 'No backend nodes available',
            timestamp: new Date().toISOString()
        });
    }

    // Build target URL
    const targetPath = req.originalUrl.replace(/^\/api/, '');
    const targetUrl = `${targetBase}${targetPath}`;
    
    // Track metrics
    registry.metrics.totalRequests++;
    
    // Set SSE headers for streaming
    if (req.path.includes('/stream') || req.path.includes('/queue')) {
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('X-Accel-Buffering', 'no');
        res.setHeader('Connection', 'keep-alive');
    }

    // Log the proxy attempt
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path} -> ${targetUrl}`);

    // Proxy the request
    proxy.web(req, res, {
        target: targetUrl,
        changeOrigin: true,
        ignorePath: true,
        followRedirects: true,
        secure: true,
        timeout: CONFIG.PROXY_TIMEOUT,
        proxyTimeout: CONFIG.PROXY_TIMEOUT
    }, (err) => {
        console.error(`Proxy error: ${err.message}`);
        registry.metrics.failedRequests++;
        
        if (!res.headersSent) {
            res.status(502).json({
                error: 'Upstream request failed',
                details: err.message,
                timestamp: new Date().toISOString()
            });
        }
    });
}

// WebSocket proxy
proxy.on('proxyReqWs', (proxyReq, req, socket, head) => {
    console.log(`WebSocket upgrade: ${req.url}`);
});

// Authentication middleware for admin routes
function requireAuth(req, res, next) {
    const token = req.headers['x-admin-token'] || req.query.token;
    
    if (token !== CONFIG.ADMIN_TOKEN) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    
    next();
}

// API Routes
app.use(express.json());

// Health check for the proxy itself
app.get('/health', (req, res) => {
    const healthy = registry.nodes.some(n => n.healthy);
    res.status(healthy ? 200 : 503).json({
        status: healthy ? 'healthy' : 'degraded',
        nodes: registry.nodes.length,
        healthyNodes: registry.nodes.filter(n => n.healthy).length,
        uptime: process.uptime(),
        memory: process.memoryUsage()
    });
});

// Public status endpoint
app.get('/_status', (req, res) => {
    const status = registry.getStatus();
    res.json({
        healthy: status.nodes.filter(n => n.healthy).length,
        total: status.nodes.length,
        policy: status.policy,
        metrics: {
            requests: status.metrics.totalRequests,
            failures: status.metrics.failedRequests
        }
    });
});

// Admin: Full status
app.get('/_admin/status', requireAuth, (req, res) => {
    res.json(registry.getStatus());
});

// Admin: Add node
app.post('/_admin/nodes', requireAuth, (req, res) => {
    const { name, baseURL, metadata } = req.body;
    
    if (!name || !baseURL) {
        return res.status(400).json({ error: 'name and baseURL required' });
    }
    
    if (!registry.validateUrl(baseURL)) {
        return res.status(400).json({ error: 'Invalid baseURL' });
    }
    
    const node = registry.addNode(name, baseURL, metadata);
    res.json({ success: true, node });
});

// Admin: Remove node
app.delete('/_admin/nodes/:id', requireAuth, (req, res) => {
    const success = registry.removeNode(req.params.id);
    
    if (!success) {
        return res.status(404).json({ error: 'Node not found' });
    }
    
    res.json({ success: true });
});

// Admin: Set active node
app.post('/_admin/active/:index', requireAuth, (req, res) => {
    const index = parseInt(req.params.index, 10);
    
    if (isNaN(index) || index < 0 || index >= registry.nodes.length) {
        return res.status(400).json({ error: 'Invalid index' });
    }
    
    registry.activeIndex = index;
    registry.save();
    res.json({ success: true, activeIndex: index });
});

// Admin: Force health check
app.post('/_admin/health-check', requireAuth, async (req, res) => {
    const checker = new HealthChecker(registry);
    await checker.checkAll();
    res.json({ success: true, nodes: registry.getStatus().nodes });
});

// Main proxy route - must be last
app.use('/api', proxyMiddleware);
app.use('/', proxyMiddleware); // Catch-all for direct proxy

// Error handling
process.on('uncaughtException', (err) => {
    console.error('Uncaught exception:', err);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled rejection at:', promise, 'reason:', reason);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('SIGTERM received, shutting down gracefully...');
    
    // Clean up WebSocket heartbeats
    if (typeof activeWebSockets !== 'undefined') {
        for (const [socket, interval] of activeWebSockets) {
            clearInterval(interval);
            try {
                socket.close();
            } catch (err) {
                // Socket might already be closed
            }
        }
        activeWebSockets.clear();
        console.log('WebSocket heartbeats cleaned up');
    }
    
    await registry.save();
    process.exit(0);
});

// Start server
async function start() {
    // Load state
    await registry.load();
    
    // Start health checker
    const healthChecker = new HealthChecker(registry);
    healthChecker.start();
    
    // Start server with WebSocket support
    const server = app.listen(CONFIG.PORT, '127.0.0.1', () => {
        console.log(`
╔════════════════════════════════════════════╗
║     Blyan Dynamic API Proxy                ║
╠════════════════════════════════════════════╣
║  Port: ${CONFIG.PORT}                              ║
║  Nodes: ${registry.nodes.length}                                ║
║  Policy: ${CONFIG.SELECTION_POLICY}                     ║
║  Health Check: ${CONFIG.HEALTH_INTERVAL_MS}ms              ║
║  Proxy Timeout: ${CONFIG.PROXY_TIMEOUT / 1000}s                   ║
╚════════════════════════════════════════════╝

Admin token: ${CONFIG.ADMIN_TOKEN}
Status: http://localhost:${CONFIG.PORT}/_status
        `);
    });

    // Handle WebSocket upgrades
    server.on('upgrade', (req, socket, head) => {
        const targetBase = registry.selectNode(req);
        
        if (!targetBase) {
            console.error('No backend for WebSocket upgrade');
            socket.destroy();
            return;
        }

        const targetUrl = targetBase + req.url;
        console.log(`[${new Date().toISOString()}] WebSocket upgrade ${req.url} -> ${targetUrl}`);

        proxy.ws(req, socket, head, {
            target: targetUrl,
            changeOrigin: true,
            ws: true,
            secure: true
        }, (err) => {
            console.error(`WebSocket proxy error: ${err.message}`);
            socket.destroy();
        });
    });

    // Per-connection WebSocket keepalive
    if (CONFIG.WS_HEARTBEAT_MS > 0) {
        proxy.on('open', (proxySocket) => {
            // Set up per-connection heartbeat
            const heartbeatInterval = setInterval(() => {
                try {
                    if (proxySocket.readyState === 1) { // WebSocket.OPEN
                        proxySocket.ping();
                    } else {
                        // Connection no longer open, clean up
                        clearInterval(heartbeatInterval);
                        activeWebSockets.delete(proxySocket);
                    }
                } catch (err) {
                    // Socket might be closed, clean up
                    clearInterval(heartbeatInterval);
                    activeWebSockets.delete(proxySocket);
                }
            }, CONFIG.WS_HEARTBEAT_MS);

            // Track the interval for cleanup
            activeWebSockets.set(proxySocket, heartbeatInterval);

            // Clean up on socket close
            proxySocket.on('close', () => {
                const interval = activeWebSockets.get(proxySocket);
                if (interval) {
                    clearInterval(interval);
                    activeWebSockets.delete(proxySocket);
                }
            });

            // Clean up on socket error
            proxySocket.on('error', () => {
                const interval = activeWebSockets.get(proxySocket);
                if (interval) {
                    clearInterval(interval);
                    activeWebSockets.delete(proxySocket);
                }
            });

            console.log(`WebSocket keepalive started (ping every ${CONFIG.WS_HEARTBEAT_MS}ms)`);
        });
    }
}

start().catch(console.error);
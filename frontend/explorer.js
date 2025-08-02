/**
 * AI-Block Explorer Dashboard
 * Comprehensive blockchain visualization and analytics
 */

class AIBlockExplorer {
    constructor() {
        this.apiBase = API_CONFIG.baseURL;
        this.data = {
            metaBlocks: [],
            parameterBlocks: [],
            expertStats: [],
            polStatus: null
        };
        this.charts = {};
        this.currentFilters = {};
        
        this.initialize();
    }

    async initialize() {
        console.log('🔍 Initializing AI-Block Explorer...');
        
        // Check API connectivity
        await this.checkAPIStatus();
        
        // Load initial data
        await this.loadAllData();
        
        // Initialize event listeners
        this.setupEventListeners();
        
        // Start auto-refresh
        this.startAutoRefresh();
        
        console.log('✅ Explorer initialized successfully');
    }

    async checkAPIStatus() {
        try {
            const response = await fetch(`${this.apiBase}/pol/status`);
            if (response.ok) {
                const status = await response.json();
                this.updateStatusIndicator('api-status', true, 'API: Online');
                this.updateStatusIndicator('pol-status', status.pol_enabled, 
                    `PoL: ${status.pol_enabled ? 'Enabled' : 'Disabled'}`);
                this.data.polStatus = status;
            } else {
                throw new Error('API not responding');
            }
        } catch (error) {
            console.error('API connection failed:', error);
            this.updateStatusIndicator('api-status', false, 'API: Offline');
            this.updateStatusIndicator('pol-status', false, 'PoL: Unknown');
        }
    }

    updateStatusIndicator(elementId, isOnline, text) {
        const element = document.getElementById(elementId);
        element.className = `status-badge ${isOnline ? 'status-online' : 'status-offline'}`;
        element.innerHTML = isOnline ? `✅ ${text}` : `❌ ${text}`;
    }

    async loadAllData() {
        console.log('📊 Loading blockchain data...');
        
        try {
            // Load data in parallel
            const [metaResponse, paramResponse, expertsResponse] = await Promise.all([
                fetch(`${this.apiBase}/chain/A/blocks`),
                fetch(`${this.apiBase}/chain/B/blocks?limit=100`),
                fetch(`${this.apiBase}/experts/top?limit=50`).catch(() => ({ json: () => ({ experts: [] }) }))
            ]);

            if (metaResponse.ok) {
                const metaData = await metaResponse.json();
                this.data.metaBlocks = metaData.blocks || [];
            }

            if (paramResponse.ok) {
                const paramData = await paramResponse.json();
                this.data.parameterBlocks = paramData.blocks || [];
            }

            if (expertsResponse.ok) {
                const expertData = await expertsResponse.json();
                this.data.expertStats = expertData.experts || [];
            }

            // Update UI
            this.updateOverviewMetrics();
            this.updatePoLMetrics();
            this.updateMetaChainView();
            this.updateParameterChainView();
            this.updateAnalyticsCharts();

            // Hide loading spinners
            this.hideLoadingSpinners();

        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load blockchain data');
        }
    }

    updateOverviewMetrics() {
        const totalBlocks = this.data.metaBlocks.length + this.data.parameterBlocks.length;
        const expertBlocks = this.data.parameterBlocks.filter(b => 
            b.block_type === 'expert' || b.block_type === 'router'
        ).length;
        
        const lastUpdate = this.data.parameterBlocks.length > 0 
            ? new Date(Math.max(...this.data.parameterBlocks.map(b => b.timestamp * 1000)))
            : new Date();

        document.getElementById('total-blocks').textContent = totalBlocks;
        document.getElementById('expert-count').textContent = expertBlocks;
        document.getElementById('last-update').textContent = this.formatRelativeTime(lastUpdate);
    }

    updatePoLMetrics() {
        if (!this.data.polStatus) {
            document.getElementById('pol-pass-rate').textContent = 'N/A';
            document.getElementById('avg-delta').textContent = 'N/A';
            document.getElementById('pol-threshold').textContent = 'N/A';
            return;
        }

        // Mock PoL metrics (in production, these would come from actual PoL results)
        const passRate = this.data.polStatus.pol_enabled ? 
            Math.round(Math.random() * 30 + 70) : 0; // 70-100% pass rate
        
        const avgDelta = this.data.polStatus.pol_enabled ? 
            (Math.random() * 0.05 + 0.01).toFixed(3) : '0.000';
        
        const threshold = this.data.polStatus.pol_threshold || 0.01;

        document.getElementById('pol-pass-rate').textContent = `${passRate}%`;
        document.getElementById('avg-delta').textContent = `+${avgDelta}`;
        document.getElementById('pol-threshold').textContent = `${(threshold * 100).toFixed(1)}%`;
    }

    updateMetaChainView() {
        const metaInfo = document.getElementById('meta-info');
        
        if (this.data.metaBlocks.length === 0) {
            metaInfo.innerHTML = '<p>No meta blocks found.</p>';
            return;
        }

        const latestMeta = this.data.metaBlocks[0];
        let metaSpec = {};
        
        try {
            // Try to decode meta block payload
            const payloadHex = latestMeta.payload || '';
            if (payloadHex) {
                const payloadBytes = this.hexToBytes(payloadHex);
                const payloadText = new TextDecoder().decode(payloadBytes);
                metaSpec = JSON.parse(payloadText);
            }
        } catch (error) {
            console.log('Could not decode meta block payload:', error);
            // Use mock data for demonstration
            metaSpec = {
                model_name: "mock-moe-model",
                architecture: "mixture-of-experts",
                num_layers: 3,
                num_experts: 4,
                routing_strategy: "top2",
                hidden_size: 512,
                expert_size: 1024
            };
        }

        metaInfo.innerHTML = `
            <div class="metric-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                <div class="metric-item">
                    <div class="metric-value">${metaSpec.model_name || 'Unknown'}</div>
                    <div class="metric-label">Model Name</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${metaSpec.architecture || 'Standard'}</div>
                    <div class="metric-label">Architecture</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${metaSpec.num_layers || 'N/A'}</div>
                    <div class="metric-label">Layers</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${metaSpec.num_experts || 'N/A'}</div>
                    <div class="metric-label">Experts per Layer</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${metaSpec.routing_strategy || 'N/A'}</div>
                    <div class="metric-label">Routing Strategy</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${metaSpec.hidden_size || 'N/A'}</div>
                    <div class="metric-label">Hidden Size</div>
                </div>
            </div>
            
            <div style="margin-top: 20px;">
                <h4>Meta Block Details</h4>
                <table class="data-table">
                    <tr>
                        <th>Block Hash</th>
                        <td class="hash-display">${latestMeta.hash}</td>
                    </tr>
                    <tr>
                        <th>Index</th>
                        <td>${latestMeta.index}</td>
                    </tr>
                    <tr>
                        <th>Timestamp</th>
                        <td>${new Date(latestMeta.timestamp * 1000).toLocaleString()}</td>
                    </tr>
                    <tr>
                        <th>Payload Size</th>
                        <td>${latestMeta.payload_size} bytes</td>
                    </tr>
                </table>
            </div>
        `;
    }

    updateParameterChainView() {
        const tableBody = document.getElementById('expert-table-body');
        
        if (this.data.parameterBlocks.length === 0) {
            tableBody.innerHTML = `
                <tr>
                    <td colspan="10" style="text-align: center; padding: 40px; color: #6b7280;">
                        No expert blocks found
                    </td>
                </tr>
            `;
            return;
        }

        // Filter expert and router blocks
        const expertBlocks = this.data.parameterBlocks.filter(block => 
            block.block_type === 'expert' || block.block_type === 'router'
        );

        // Populate layer filter
        const layers = [...new Set(expertBlocks.map(b => b.layer_id).filter(Boolean))];
        const layerSelect = document.getElementById('layer-filter');
        layerSelect.innerHTML = '<option value="">All Layers</option>' + 
            layers.map(layer => `<option value="${layer}">${layer}</option>`).join('');

        // Create table rows
        const rows = expertBlocks.map(block => {
            const expertStats = this.data.expertStats.find(stat => 
                stat.expert_name === block.expert_name
            );
            
            const usageCount = expertStats ? expertStats.call_count : 0;
            const deltaScore = this.generateMockDeltaScore(); // Mock delta score
            const status = usageCount > 0 ? 'active' : 'inactive';
            
            return `
                <tr data-expert="${block.expert_name}" data-layer="${block.layer_id}" 
                    data-type="${block.block_type}" data-status="${status}">
                    <td>
                        <strong>${block.expert_name || 'Unknown'}</strong>
                    </td>
                    <td>${block.layer_id || 'N/A'}</td>
                    <td>
                        <span class="score-badge ${block.block_type === 'expert' ? 'score-good' : 'score-fair'}">
                            ${block.block_type || 'unknown'}
                        </span>
                    </td>
                    <td>${(block.payload_size / 1024).toFixed(1)}</td>
                    <td class="hash-display">${block.hash}</td>
                    <td>${block.index}</td>
                    <td>${usageCount}</td>
                    <td>
                        <span class="score-badge ${this.getDeltaScoreBadge(deltaScore)}">
                            ${deltaScore >= 0 ? '+' : ''}${deltaScore.toFixed(3)}
                        </span>
                    </td>
                    <td>
                        <div class="expert-status">
                            <span class="status-dot status-${status}"></span>
                            ${status}
                        </div>
                    </td>
                    <td>
                        <button class="btn btn-primary" style="font-size: 12px; padding: 4px 8px;" 
                            onclick="viewBlockDetails('${block.hash}')">
                            View
                        </button>
                    </td>
                </tr>
            `;
        }).join('');

        tableBody.innerHTML = rows;
    }

    updateAnalyticsCharts() {
        // Create timeline chart
        this.createBlocksTimelineChart();
        
        // Create improvement trend chart
        this.createImprovementTrendChart();
        
        // Create expert usage chart
        this.createExpertUsageChart();
        
        // Create PoL metrics chart
        this.createPoLMetricsChart();
    }

    createBlocksTimelineChart() {
        const ctx = document.getElementById('blocks-timeline-chart');
        if (!ctx) return;

        // Generate timeline data
        const timelineData = this.generateTimelineData();
        
        if (this.charts.timeline) {
            this.charts.timeline.destroy();
        }

        this.charts.timeline = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timelineData.labels,
                datasets: [{
                    label: 'Expert Blocks',
                    data: timelineData.expertBlocks,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Router Blocks',
                    data: timelineData.routerBlocks,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Block Creation Timeline'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    createImprovementTrendChart() {
        const ctx = document.getElementById('improvement-trend-chart');
        if (!ctx) return;

        // Generate improvement trend data
        const trendData = this.generateImprovementTrendData();
        
        if (this.charts.improvement) {
            this.charts.improvement.destroy();
        }

        this.charts.improvement = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: trendData.labels,
                datasets: [{
                    label: 'Average Δ Score',
                    data: trendData.scores,
                    backgroundColor: trendData.scores.map(score => 
                        score >= 0.02 ? '#10b981' : 
                        score >= 0.01 ? '#3b82f6' : 
                        score >= 0.005 ? '#f59e0b' : '#ef4444'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Improvement Score Trends'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Delta Score'
                        }
                    }
                }
            }
        });
    }

    createExpertUsageChart() {
        const ctx = document.getElementById('expert-usage-chart');
        if (!ctx) return;

        const topExperts = this.data.expertStats.slice(0, 10);
        
        if (this.charts.usage) {
            this.charts.usage.destroy();
        }

        this.charts.usage = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: topExperts.map(expert => expert.expert_name),
                datasets: [{
                    data: topExperts.map(expert => expert.call_count),
                    backgroundColor: [
                        '#2563eb', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
                        '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6b7280'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Expert Usage Distribution'
                    },
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    createPoLMetricsChart() {
        const ctx = document.getElementById('pol-metrics-chart');
        if (!ctx) return;

        // Generate PoL metrics data
        const polData = this.generatePoLMetricsData();
        
        if (this.charts.pol) {
            this.charts.pol.destroy();
        }

        this.charts.pol = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Pass Rate', 'Avg Delta', 'Quality Score', 'Speed', 'Reliability'],
                datasets: [{
                    label: 'PoL Performance',
                    data: polData,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.2)',
                    pointBackgroundColor: '#2563eb'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'PoL System Performance'
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    // Utility functions
    generateTimelineData() {
        const days = 7;
        const labels = [];
        const expertBlocks = [];
        const routerBlocks = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString());
            
            // Mock data - in production, this would be calculated from actual block timestamps
            expertBlocks.push(Math.floor(Math.random() * 5) + 1);
            routerBlocks.push(Math.floor(Math.random() * 2));
        }
        
        return { labels, expertBlocks, routerBlocks };
    }

    generateImprovementTrendData() {
        const experts = this.data.parameterBlocks
            .filter(b => b.expert_name)
            .slice(0, 10);
        
        return {
            labels: experts.map(b => b.expert_name),
            scores: experts.map(() => this.generateMockDeltaScore())
        };
    }

    generatePoLMetricsData() {
        if (!this.data.polStatus || !this.data.polStatus.pol_enabled) {
            return [0, 0, 0, 0, 0];
        }
        
        // Mock PoL performance metrics (0-100 scale)
        return [
            Math.random() * 30 + 70, // Pass rate
            Math.random() * 40 + 60, // Avg delta
            Math.random() * 25 + 75, // Quality score
            Math.random() * 20 + 80, // Speed
            Math.random() * 15 + 85  // Reliability
        ];
    }

    generateMockDeltaScore() {
        // Generate realistic delta scores
        const random = Math.random();
        if (random < 0.7) {
            return Math.random() * 0.05 + 0.005; // Positive improvement
        } else {
            return Math.random() * -0.02; // Some negative scores
        }
    }

    getDeltaScoreBadge(score) {
        if (score >= 0.02) return 'score-excellent';
        if (score >= 0.01) return 'score-good';
        if (score >= 0.005) return 'score-fair';
        return 'score-poor';
    }

    formatRelativeTime(date) {
        const now = new Date();
        const diffMs = now - date;
        const diffMinutes = Math.floor(diffMs / 60000);
        
        if (diffMinutes < 1) return 'Just now';
        if (diffMinutes < 60) return `${diffMinutes}m ago`;
        
        const diffHours = Math.floor(diffMinutes / 60);
        if (diffHours < 24) return `${diffHours}h ago`;
        
        const diffDays = Math.floor(diffHours / 24);
        return `${diffDays}d ago`;
    }

    hexToBytes(hex) {
        const bytes = new Uint8Array(hex.length / 2);
        for (let i = 0; i < hex.length; i += 2) {
            bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
        }
        return bytes;
    }

    hideLoadingSpinners() {
        const spinners = ['overview-loading', 'pol-loading', 'meta-loading', 'param-loading', 'analytics-loading'];
        spinners.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.style.display = 'none';
        });
    }

    setupEventListeners() {
        // Search and filter inputs
        document.getElementById('expert-search').addEventListener('input', 
            this.debounce(() => this.applyFilters(), 300));
        
        document.getElementById('layer-filter').addEventListener('change', 
            () => this.applyFilters());
        
        document.getElementById('type-filter').addEventListener('change', 
            () => this.applyFilters());
        
        document.getElementById('status-filter').addEventListener('change', 
            () => this.applyFilters());
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    startAutoRefresh() {
        // Refresh data every 30 seconds
        setInterval(() => {
            this.loadAllData();
        }, 30000);
    }

    showError(message) {
        console.error(message);
        // Could implement toast notifications here
    }
}

// Global functions - 이 함수는 common-header.js에서 처리되므로 제거
// function switchTab(tabId) {
//     // Hide all tab contents
//     document.querySelectorAll('.tab-content').forEach(tab => {
//         tab.classList.remove('active');
//     });
//     
//     // Remove active class from all tab buttons
//     document.querySelectorAll('.tab-button').forEach(button => {
//         button.classList.remove('active');
//     });
//     
//     // Show selected tab
//     document.getElementById(tabId).classList.add('active');
//     event.target.classList.add('active');
//     
//     // Load DAG visualization if DAG tab is selected
//     if (tabId === 'dag-view') {
//         window.explorer.initializeDAGView();
//     }
// }

function applyFilters() {
    window.explorer.applyFilters();
}

function clearFilters() {
    document.getElementById('expert-search').value = '';
    document.getElementById('layer-filter').value = '';
    document.getElementById('type-filter').value = '';
    document.getElementById('status-filter').value = '';
    window.explorer.applyFilters();
}

function refreshAllData() {
    const button = document.querySelector('.refresh-button');
    button.innerHTML = '⏳';
    
    window.explorer.loadAllData().then(() => {
        button.innerHTML = '🔄';
    });
}

function viewBlockDetails(blockHash) {
    // Open PoL validator with the specific block
    window.open(`pol_validator.html?hash=${blockHash}`, '_blank');
}

function resetDAGView() {
    window.explorer.resetDAGView();
}

function refreshDAG() {
    window.explorer.refreshDAG();
}

// Explorer extension for additional features
AIBlockExplorer.prototype.applyFilters = function() {
    const searchTerm = document.getElementById('expert-search').value.toLowerCase();
    const layerFilter = document.getElementById('layer-filter').value;
    const typeFilter = document.getElementById('type-filter').value;
    const statusFilter = document.getElementById('status-filter').value;
    
    const rows = document.querySelectorAll('#expert-table-body tr[data-expert]');
    
    rows.forEach(row => {
        const expertName = row.dataset.expert.toLowerCase();
        const layer = row.dataset.layer;
        const type = row.dataset.type;
        const status = row.dataset.status;
        
        const matchesSearch = !searchTerm || expertName.includes(searchTerm);
        const matchesLayer = !layerFilter || layer === layerFilter;
        const matchesType = !typeFilter || type === typeFilter;
        const matchesStatus = !statusFilter || status === statusFilter;
        
        if (matchesSearch && matchesLayer && matchesType && matchesStatus) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
};

AIBlockExplorer.prototype.initializeDAGView = function() {
    const container = document.getElementById('dag-container');
    
    // Clear previous content
    container.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #6b7280;">
            <div style="text-align: center;">
                <div class="loading-spinner" style="margin: 0 auto 15px;"></div>
                <p>Loading DAG structure...</p>
                <p style="font-size: 12px;">This feature visualizes the blockchain DAG using D3.js</p>
            </div>
        </div>
    `;
    
    // Initialize D3 DAG visualization
    setTimeout(() => {
        this.renderDAGVisualization();
    }, 1000);
};

AIBlockExplorer.prototype.renderDAGVisualization = function() {
    const container = document.getElementById('dag-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Clear container
    container.innerHTML = '';
    
    // Create SVG
    const svg = d3.select('#dag-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Create sample DAG data
    const nodes = [
        { id: 'meta-0', type: 'meta', name: 'Meta Block', x: width/2, y: 50 },
        ...this.data.parameterBlocks.slice(0, 20).map((block, i) => ({
            id: block.hash,
            type: block.block_type || 'expert',
            name: block.expert_name || `Block ${i}`,
            x: Math.random() * (width - 100) + 50,
            y: Math.random() * (height - 200) + 100
        }))
    ];
    
    const links = nodes.slice(1).map(node => ({
        source: 'meta-0',
        target: node.id
    }));
    
    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2));
    
    // Add links
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter().append('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', 2);
    
    // Add nodes
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('r', d => d.type === 'meta' ? 15 : 10)
        .attr('fill', d => {
            switch(d.type) {
                case 'meta': return '#2563eb';
                case 'expert': return '#10b981';
                case 'router': return '#f59e0b';
                default: return '#6b7280';
            }
        })
        .call(d3.drag()
            .on('start', (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }));
    
    // Add labels
    const label = svg.append('g')
        .selectAll('text')
        .data(nodes)
        .enter().append('text')
        .text(d => d.name)
        .attr('font-size', '10px')
        .attr('text-anchor', 'middle')
        .attr('dy', 25);
    
    // Update positions on simulation tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
};

AIBlockExplorer.prototype.resetDAGView = function() {
    this.initializeDAGView();
};

AIBlockExplorer.prototype.refreshDAG = function() {
    this.loadAllData().then(() => {
        if (document.getElementById('dag-view').classList.contains('active')) {
            this.initializeDAGView();
        }
    });
};

// Initialize explorer when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.explorer = new AIBlockExplorer();
});
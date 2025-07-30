/**
 * PoL Block Validator UI Logic
 */

class PoLValidator {
    constructor() {
        this.apiBase = 'http://127.0.0.1:8000';
        this.selectedBlock = null;
        
        // 페이지 로드 후 초기화
        setTimeout(() => {
            this.initializeElements();
            this.attachEventListeners();
            this.checkPoLStatus();
        }, 200);
    }
    
    initializeElements() {
        // Selection elements
        this.selectionMethod = document.getElementById('selection-method');
        this.hashInputSection = document.getElementById('hash-input-section');
        this.blockListSection = document.getElementById('block-list-section');
        this.blockHashInput = document.getElementById('block-hash-input');
        this.blocksList = document.getElementById('blocks-list');
        this.loadBlockBtn = document.getElementById('load-block-btn');
        
        // Block info elements
        this.blockInfoSection = document.getElementById('block-info-section');
        this.blockExpertName = document.getElementById('block-expert-name');
        this.blockLayerId = document.getElementById('block-layer-id');
        this.blockHashDisplay = document.getElementById('block-hash-display');
        this.blockType = document.getElementById('block-type');
        this.blockPayloadSize = document.getElementById('block-payload-size');
        this.blockTimestamp = document.getElementById('block-timestamp');
        
        // Validation elements
        this.validateBtn = document.getElementById('validate-btn');
        this.validationLoading = document.getElementById('validation-loading');
        this.validationResults = document.getElementById('validation-results');
        
        // Admin elements
        this.adminSection = document.getElementById('admin-section');
        this.adminAcceptBtn = document.getElementById('admin-accept-btn');
        this.adminRejectBtn = document.getElementById('admin-reject-btn');
        this.adminReason = document.getElementById('admin-reason');
        this.adminResults = document.getElementById('admin-results');
    }
    
    attachEventListeners() {
        if (this.selectionMethod) {
            this.selectionMethod.addEventListener('change', () => this.toggleSelectionMethod());
        }
        
        if (this.loadBlockBtn) {
            this.loadBlockBtn.addEventListener('click', () => this.loadBlockInfo());
        }
        
        if (this.validateBtn) {
            this.validateBtn.addEventListener('click', () => this.runValidation());
        }
        
        if (this.adminAcceptBtn) {
            this.adminAcceptBtn.addEventListener('click', () => this.manualAccept());
        }
        
        if (this.adminRejectBtn) {
            this.adminRejectBtn.addEventListener('click', () => this.manualReject());
        }
        
        // Hash input validation
        if (this.blockHashInput) {
            this.blockHashInput.addEventListener('input', () => this.validateHashInput());
        }
    }
    
    async checkPoLStatus() {
        try {
            const response = await fetch(`${this.apiBase}/pol/status`);
            const statusData = await response.json();
            
            if (!statusData.pol_enabled) {
                this.showError('PoL system is not enabled on the server.');
                return;
            }
            
            console.log('PoL Status:', statusData);
            this.loadRecentBlocks();
        } catch (error) {
            this.showError(`Could not connect to API server: ${error.message}`);
        }
    }
    
    toggleSelectionMethod() {
        if (!this.selectionMethod) return;
        
        const method = this.selectionMethod.value;
        
        if (method === 'hash') {
            if (this.hashInputSection) this.hashInputSection.style.display = 'block';
            if (this.blockListSection) this.blockListSection.style.display = 'none';
        } else {
            if (this.hashInputSection) this.hashInputSection.style.display = 'none';
            if (this.blockListSection) this.blockListSection.style.display = 'block';
            this.loadRecentBlocks();
        }
        
        this.clearBlockInfo();
    }
    
    validateHashInput() {
        if (!this.blockHashInput || !this.loadBlockBtn) return;
        
        const hash = this.blockHashInput.value;
        const isValid = /^[a-fA-F0-9]{64}$/.test(hash);
        
        this.loadBlockBtn.disabled = !isValid;
        
        if (hash.length > 0 && !isValid) {
            this.blockHashInput.style.borderColor = '#dc3545';
        } else {
            this.blockHashInput.style.borderColor = '#ddd';
        }
    }
    
    async loadRecentBlocks() {
        if (!this.blocksList) return;
        
        try {
            this.blocksList.innerHTML = '<div style="padding: 20px; text-align: center;">Loading blocks...</div>';
            
            const response = await fetch(`${this.apiBase}/chain/B/blocks?limit=20`);
            const data = await response.json();
            
            const expertBlocks = data.blocks.filter(block => 
                block.block_type === 'expert' || block.block_type === 'router'
            );
            
            if (expertBlocks.length === 0) {
                this.blocksList.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">No expert blocks found</div>';
                return;
            }
            
            this.blocksList.innerHTML = '';
            
            expertBlocks.forEach(block => {
                const blockItem = document.createElement('div');
                blockItem.className = 'block-item';
                blockItem.style.cssText = `
                    padding: 15px;
                    margin: 10px 0;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                `;
                
                blockItem.innerHTML = `
                    <div>
                        <strong>${block.expert_name || 'Unknown Expert'}</strong>
                        <span style="color: #666; margin-left: 10px;">${block.block_type}</span>
                    </div>
                    <div style="font-size: 12px; color: #999; margin-top: 5px;">
                        Hash: ${block.hash.substring(0, 16)}...
                    </div>
                    <div style="font-size: 12px; color: #999;">
                        Layer: ${block.layer_id || 'N/A'} | 
                        Size: ${block.payload_size} bytes | 
                        ${new Date(block.timestamp * 1000).toLocaleString()}
                    </div>
                `;
                
                blockItem.addEventListener('click', () => {
                    this.selectBlockFromList(block);
                });
                
                blockItem.addEventListener('mouseenter', () => {
                    blockItem.style.backgroundColor = '#f5f5f5';
                });
                
                blockItem.addEventListener('mouseleave', () => {
                    if (this.selectedBlock !== block) {
                        blockItem.style.backgroundColor = '';
                    }
                });
                
                this.blocksList.appendChild(blockItem);
            });
            
        } catch (error) {
            this.blocksList.innerHTML = `<div style="padding: 20px; text-align: center; color: #dc3545;">Error loading blocks: ${error.message}</div>`;
        }
    }
    
    selectBlockFromList(block) {
        // Remove previous selection
        document.querySelectorAll('.block-item').forEach(item => {
            item.style.backgroundColor = '';
        });
        
        // Highlight selected block
        event.currentTarget.style.backgroundColor = '#e3f2fd';
        
        this.selectedBlock = block;
        this.displayBlockInfo(block);
        
        // 자동으로 Block Info 탭으로 전환
        if (typeof switchTab === 'function') {
            switchTab('block-info');
        }
    }
    
    async loadBlockInfo() {
        if (!this.selectionMethod) return;
        
        const method = this.selectionMethod.value;
        
        if (method === 'hash') {
            const hash = this.blockHashInput.value;
            if (!/^[a-fA-F0-9]{64}$/.test(hash)) {
                this.showError('Please enter a valid 64-character block hash.');
                return;
            }
            
            try {
                const response = await fetch(`${this.apiBase}/chain/B/blocks?limit=100`);
                const data = await response.json();
                
                const block = data.blocks.find(b => b.hash === hash);
                if (!block) {
                    this.showError('Block not found with the specified hash.');
                    return;
                }
                
                this.selectedBlock = block;
                this.displayBlockInfo(block);
                
                // 자동으로 Block Info 탭으로 전환
                if (typeof switchTab === 'function') {
                    switchTab('block-info');
                }
                
            } catch (error) {
                this.showError(`Error loading block: ${error.message}`);
            }
        }
    }
    
    displayBlockInfo(block) {
        if (this.blockExpertName) this.blockExpertName.textContent = block.expert_name || 'N/A';
        if (this.blockLayerId) this.blockLayerId.textContent = block.layer_id || 'N/A';
        if (this.blockHashDisplay) this.blockHashDisplay.textContent = block.hash;
        if (this.blockType) this.blockType.textContent = block.block_type || 'N/A';
        if (this.blockPayloadSize) this.blockPayloadSize.textContent = `${block.payload_size} bytes`;
        if (this.blockTimestamp) this.blockTimestamp.textContent = new Date(block.timestamp * 1000).toLocaleString();
        
        if (this.blockInfoSection) this.blockInfoSection.style.display = 'block';
        if (this.validateBtn) this.validateBtn.disabled = false;
        
        // Show admin section for expert blocks
        if ((block.block_type === 'expert' || block.block_type === 'router') && this.adminSection) {
            this.adminSection.style.display = 'block';
        }
    }
    
    clearBlockInfo() {
        if (this.blockInfoSection) this.blockInfoSection.style.display = 'none';
        if (this.validateBtn) this.validateBtn.disabled = true;
        if (this.validationResults) this.validationResults.style.display = 'none';
        if (this.adminSection) this.adminSection.style.display = 'none';
        this.selectedBlock = null;
    }
    
    async runValidation() {
        if (!this.selectedBlock) {
            this.showError('No block selected for validation.');
            return;
        }
        
        if (this.validateBtn) this.validateBtn.disabled = true;
        if (this.validationLoading) this.validationLoading.style.display = 'block';
        if (this.validationResults) this.validationResults.style.display = 'none';
        
        try {
            const request = {
                expert_name: this.selectedBlock.expert_name,
                layer_id: this.selectedBlock.layer_id,
                block_hash: this.selectedBlock.hash
            };
            
            const response = await fetch(`${this.apiBase}/pol/validate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(request)
            });
            
            const result = await response.json();
            
            if (this.validationLoading) this.validationLoading.style.display = 'none';
            if (this.validateBtn) this.validateBtn.disabled = false;
            
            if (response.ok) {
                this.displayValidationResults(result);
            } else {
                this.showValidationError(result.detail || 'Validation failed');
            }
            
        } catch (error) {
            if (this.validationLoading) this.validationLoading.style.display = 'none';
            if (this.validateBtn) this.validateBtn.disabled = false;
            this.showValidationError(`Network error: ${error.message}`);
        }
    }
    
    displayValidationResults(result) {
        if (!this.validationResults) return;
        
        const isValid = result.is_valid;
        
        let resultsHTML = `
            <h3>Validation Results</h3>
            <div style="margin-bottom: 15px;">
                <strong>Status:</strong> 
                <span style="padding: 5px 10px; border-radius: 4px; color: white; background-color: ${isValid ? '#10b981' : '#ef4444'};">
                    ${isValid ? '✅ VALID' : '❌ INVALID'}
                </span>
            </div>
        `;
        
        if (result.validation_details && result.validation_details.pol_metrics) {
            const metrics = result.validation_details.pol_metrics;
            
            resultsHTML += `
                <h4>PoL Metrics</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Candidate Score</th><td style="padding: 8px; border-bottom: 1px solid #ddd;">${metrics.candidate_score?.toFixed(4) || 'N/A'}</td></tr>
                    <tr><th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Baseline Score</th><td style="padding: 8px; border-bottom: 1px solid #ddd;">${metrics.baseline_score?.toFixed(4) || 'N/A'}</td></tr>
                    <tr><th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Delta Score</th><td style="padding: 8px; border-bottom: 1px solid #ddd;">${metrics.delta_score?.toFixed(4) || 'N/A'}</td></tr>
                    <tr><th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Improvement</th><td style="padding: 8px; border-bottom: 1px solid #ddd;">${metrics.improvement_percentage?.toFixed(2) || '0'}%</td></tr>
                    <tr><th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Meets Threshold</th><td style="padding: 8px; border-bottom: 1px solid #ddd;">${metrics.meets_threshold ? '✅ Yes' : '❌ No'}</td></tr>
                </table>
            `;
        }
        
        this.validationResults.innerHTML = resultsHTML;
        this.validationResults.style.display = 'block';
        this.validationResults.style.padding = '20px';
        this.validationResults.style.backgroundColor = isValid ? '#f0f9ff' : '#fef2f2';
        this.validationResults.style.border = `1px solid ${isValid ? '#0ea5e9' : '#f87171'}`;
        this.validationResults.style.borderRadius = '6px';
    }
    
    showValidationError(message) {
        if (!this.validationResults) return;
        
        this.validationResults.innerHTML = `
            <h3>Validation Error</h3>
            <p>${message}</p>
        `;
        this.validationResults.style.display = 'block';
        this.validationResults.style.padding = '20px';
        this.validationResults.style.backgroundColor = '#fef2f2';
        this.validationResults.style.border = '1px solid #f87171';
        this.validationResults.style.borderRadius = '6px';
    }
    
    async manualAccept() {
        this.performAdminAction('accept');
    }
    
    async manualReject() {
        this.performAdminAction('reject');
    }
    
    performAdminAction(action) {
        if (!this.selectedBlock) {
            this.showAdminError('No block selected.');
            return;
        }
        
        const reason = this.adminReason ? this.adminReason.value.trim() : '';
        
        if (this.adminResults) {
            this.adminResults.innerHTML = `
                <h4>Admin Action: ${action.toUpperCase()}</h4>
                <p><strong>Block:</strong> ${this.selectedBlock.hash.substring(0, 16)}...</p>
                <p><strong>Expert:</strong> ${this.selectedBlock.expert_name}</p>
                ${reason ? `<p><strong>Reason:</strong> ${reason}</p>` : ''}
                <p><strong>Status:</strong> <span style="color: #10b981; font-weight: bold;">Action Recorded</span></p>
                <p><em>Note: This is a demo action.</em></p>
            `;
            
            this.adminResults.style.display = 'block';
            this.adminResults.style.padding = '15px';
            this.adminResults.style.backgroundColor = '#f0f9ff';
            this.adminResults.style.border = '1px solid #0ea5e9';
            this.adminResults.style.borderRadius = '6px';
        }
        
        // Clear reason input
        if (this.adminReason) this.adminReason.value = '';
    }
    
    showAdminError(message) {
        if (this.adminResults) {
            this.adminResults.innerHTML = `<p>Error: ${message}</p>`;
            this.adminResults.style.display = 'block';
            this.adminResults.style.backgroundColor = '#fef2f2';
            this.adminResults.style.border = '1px solid #f87171';
            this.adminResults.style.borderRadius = '6px';
        }
    }
    
    showError(message) {
        // Create temporary error display
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = `<p><strong>Error:</strong> ${message}</p>`;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 400px;
            padding: 15px;
            background-color: #fef2f2;
            border: 1px solid #f87171;
            border-radius: 6px;
            color: #991b1b;
        `;
        
        document.body.appendChild(errorDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }
}

// Initialize validator when page loads
document.addEventListener('DOMContentLoaded', () => {
    new PoLValidator();
});
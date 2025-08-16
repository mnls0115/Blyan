/**
 * Blyan 공통 헤더 컴포넌트
 */

function createAIBlockHeader(currentPage = '') {
    const pages = {
        'home': { icon: '🏠', nameKey: 'home', url: 'home.html' },
        'chat': { icon: '💬', nameKey: 'chat', url: 'chat.html' },
        'contribute': { icon: '⚡', nameKey: 'network', url: 'contribute.html' },
        // Temporarily hidden - will be re-enabled later
        // 'leaderboard': { icon: '🏅', nameKey: 'leaderboard', url: 'leaderboard.html' },
        // 'explorer': { icon: '🔧', nameKey: 'technical', url: 'explorer.html' }
    };

    return `
        <header class="Blyan-header">
            <div class="Blyan-nav">
                <div class="Blyan-title">
                    🤖 Blyan
                </div>
                <nav class="nav-tabs">
                    ${Object.entries(pages).map(([key, page]) => `
                        <a href="${page.url}" class="${currentPage === key ? 'active' : ''}">
                            ${page.icon} <span data-i18n="${page.nameKey}">${t(page.nameKey)}</span>
                        </a>
                    `).join('')}
                </nav>
                <div class="status-indicators">
                    ${typeof createLanguageSelector !== 'undefined' ? createLanguageSelector() : ''}
                    <div id="header-usage-indicator" class="header-usage-badge" style="display: none;">
                        <span id="header-usage-text">--</span>
                    </div>
                    <!-- Wallet functionality hidden for MVP - can be restored later -->
                    <!--
                    <div id="wallet-info" style="display: none;">
                        <span id="wallet-balance" class="wallet-balance">0 BLY</span>
                        <span id="wallet-address" class="wallet-address"></span>
                        <span id="contributor-badge" class="contributor-badge"></span>
                    </div>
                    <button id="wallet-connect-btn" class="wallet-button" data-i18n="connectWallet">${typeof t !== 'undefined' ? t('connectWallet') : 'Connect Wallet'}</button>
                    -->
                    <div id="api-status" class="status-badge status-offline">
                        <span data-i18n="apiStatus">${typeof t !== 'undefined' ? t('apiStatus') : 'API'}</span>: <span data-i18n="offline">${typeof t !== 'undefined' ? t('offline') : 'Offline'}</span>
                    </div>
                    <div id="pol-status" class="status-badge status-offline">
                        <span data-i18n="polStatus">${typeof t !== 'undefined' ? t('polStatus') : 'PoL'}</span>: <span data-i18n="unknown">${typeof t !== 'undefined' ? t('unknown') : 'Unknown'}</span>
                    </div>
                </div>
            </div>
        </header>
    `;
}

// Function to refresh header (called when language changes)
function refreshHeader() {
    const currentPageElement = document.querySelector('.nav-tabs a.active');
    let currentPage = '';
    if (currentPageElement) {
        const href = currentPageElement.getAttribute('href');
        if (href.includes('home.html')) currentPage = 'home';
        else if (href.includes('chat.html')) currentPage = 'chat';
        else if (href.includes('contribute.html')) currentPage = 'contribute';
        else if (href.includes('leaderboard.html')) currentPage = 'leaderboard';
        else if (href.includes('explorer.html')) currentPage = 'explorer';
    }
    
    const headerContainer = document.getElementById('header-container');
    if (headerContainer) {
        headerContainer.innerHTML = createAIBlockHeader(currentPage);
        updatePageLanguage();
    }
}

function createPageTabs(tabs, activeTab = '') {
    const tabsHtml = `
        <div class="page-tabs" style="display: flex !important; flex-direction: row !important; width: 100%;">
            ${tabs.map(tab => `
                <button class="page-tab ${activeTab === tab.id ? 'active' : ''}" 
                        onclick="switchTab('${tab.id}')">
                    ${tab.icon} ${tab.nameKey ? `<span data-i18n="${tab.nameKey}">${typeof t !== 'undefined' ? t(tab.nameKey) : tab.nameKey}</span>` : (tab.name || '')}
                </button>
            `).join('')}
        </div>
    `;
    console.log('Created tabs HTML:', tabsHtml);
    return tabsHtml;
}

// 탭 전환 함수
function switchTab(tabId) {
    console.log('Switching to tab:', tabId);
    
    // 모든 탭 버튼에서 active 제거
    document.querySelectorAll('.page-tab').forEach(tab => {
        tab.classList.remove('active');
        console.log('Removed active from:', tab.textContent);
    });
    
    // 모든 탭 컨텐츠 숨기기
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
        console.log('Hidden content:', content.id);
    });
    
    // 선택된 탭 버튼 활성화
    const selectedTab = document.querySelector(`[onclick="switchTab('${tabId}')"]`);
    if (selectedTab) {
        selectedTab.classList.add('active');
        console.log('Activated tab button:', selectedTab.textContent);
    }
    
    // 선택된 탭 컨텐츠 표시
    const selectedContent = document.getElementById(tabId);
    if (selectedContent) {
        selectedContent.classList.add('active');
        selectedContent.style.display = 'block';
        console.log('Shown content:', tabId);
    }
    
    // DAG 뷰 초기화 (해당 탭인 경우)
    if (tabId === 'dag-view' && window.explorer) {
        window.explorer.initializeDAGView();
    }
}

// API 상태 체크 함수
async function checkAPIStatus() {
    try {
        const response = await fetch(API_CONFIG.baseURL + API_CONFIG.polStatus);
        if (response.ok) {
            const status = await response.json();
            updateStatusBadge('api-status', true, 'API: Online');
            updateStatusBadge('pol-status', status.pol_enabled, 
                `PoL: ${status.pol_enabled ? 'Enabled' : 'Disabled'}`);
        } else {
            throw new Error('API not responding');
        }
    } catch (error) {
        updateStatusBadge('api-status', false, 'API: Offline');
        updateStatusBadge('pol-status', false, 'PoL: Unknown');
    }
}

function updateStatusBadge(elementId, isOnline, text) {
    const element = document.getElementById(elementId);
    if (element) {
        // Parse the text to use translations
        if (elementId === 'api-status') {
            element.innerHTML = `<span data-i18n="apiStatus">${typeof t !== 'undefined' ? t('apiStatus') : 'API'}</span>: <span data-i18n="${isOnline ? 'online' : 'offline'}">${typeof t !== 'undefined' ? t(isOnline ? 'online' : 'offline') : (isOnline ? 'Online' : 'Offline')}</span>`;
        } else if (elementId === 'pol-status') {
            const statusKey = text.includes('Enabled') ? 'enabled' : text.includes('Disabled') ? 'disabled' : 'unknown';
            element.innerHTML = `<span data-i18n="polStatus">${typeof t !== 'undefined' ? t('polStatus') : 'PoL'}</span>: <span data-i18n="${statusKey}">${typeof t !== 'undefined' ? t(statusKey) : text.split(': ')[1]}</span>`;
        } else {
            element.textContent = text;
        }
        element.className = `status-badge ${isOnline ? 'status-online' : 'status-offline'}`;
    }
}

// Global function to update header usage indicator
async function updateHeaderUsage() {
    const headerIndicator = document.getElementById('header-usage-indicator');
    const headerText = document.getElementById('header-usage-text');
    
    if (!headerIndicator || !headerText) return;
    
    try {
        const userAddress = getUserAddress();
        const response = await fetch(`${API_CONFIG.baseURL}/leaderboard/me/summary?address=${userAddress}`);
        
        let data;
        if (response.ok) {
            data = await response.json();
        } else if (response.status === 404) {
            // New user
            data = { free_requests_remaining: 5, balance: 0, is_new_user: true };
        } else {
            return; // Don't show on error
        }
        
        const freeRemaining = data.free_requests_remaining || 0;
        const balance = parseFloat(data.balance || 0);
        
        // Reset classes
        headerIndicator.className = 'header-usage-badge';
        
        if (freeRemaining > 0) {
            headerIndicator.classList.add('free-tier');
            headerText.textContent = `🆓 ${freeRemaining} free`;
        } else if (balance > 0) {
            if (balance < 0.010) {
                headerIndicator.classList.add('low-balance');
                headerText.textContent = `⚠️ ${balance.toFixed(4)} BLY`;
            } else {
                headerIndicator.classList.add('paid-tier');
                headerText.textContent = `💰 ${balance.toFixed(4)} BLY`;
            }
        } else {
            headerIndicator.classList.add('low-balance');
            headerText.textContent = '❌ No credits';
        }
        
        headerIndicator.style.display = 'block';
        
    } catch (error) {
        console.error('Error updating header usage:', error);
    }
}

function getUserAddress() {
    return localStorage.getItem('userAddress') || '0x' + Math.random().toString(16).substr(2, 40);
}

// Make functions globally available
window.updateHeaderUsage = updateHeaderUsage;
window.getUserAddress = getUserAddress;

// 페이지 로드시 API 상태 체크
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    setInterval(checkAPIStatus, 30000); // 30초마다 체크
});
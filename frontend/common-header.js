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
                    <img src="/logo/Blyan%20logo.png" alt="Blyan" style="height: 32px; width: auto; vertical-align: middle; margin-right: 8px;" />
                    Blyan
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
        // 헤더 재생성 후 캐시된 API 상태 로드
        loadCachedAPIStatus();
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
            const apiStatus = { api: true, pol: status.pol_enabled };
            localStorage.setItem('apiStatus', JSON.stringify(apiStatus));
            updateStatusBadge('api-status', true, 'API: Online');
            updateStatusBadge('pol-status', status.pol_enabled, 
                `PoL: ${status.pol_enabled ? 'Enabled' : 'Disabled'}`);
        } else {
            throw new Error('API not responding');
        }
    } catch (error) {
        const apiStatus = { api: false, pol: false };
        localStorage.setItem('apiStatus', JSON.stringify(apiStatus));
        updateStatusBadge('api-status', false, 'API: Offline');
        updateStatusBadge('pol-status', false, 'PoL: Unknown');
    }
}

// 캐시된 API 상태 로드 함수
function loadCachedAPIStatus() {
    try {
        const cached = localStorage.getItem('apiStatus');
        if (cached) {
            const status = JSON.parse(cached);
            updateStatusBadge('api-status', status.api, status.api ? 'API: Online' : 'API: Offline');
            updateStatusBadge('pol-status', status.pol, 
                status.pol ? 'PoL: Enabled' : status.api ? 'PoL: Disabled' : 'PoL: Unknown');
        }
    } catch (error) {
        // 캐시 로드 실패 시 기본값 사용
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

// Header usage functionality removed - moved to chat page

function getUserAddress() {
    return localStorage.getItem('userAddress') || '0x' + Math.random().toString(16).substr(2, 40);
}

// Make functions globally available
window.getUserAddress = getUserAddress;

// 페이지 로드시 API 상태 체크
document.addEventListener('DOMContentLoaded', () => {
    // 캐시된 상태를 먼저 로드
    loadCachedAPIStatus();
    // 실제 API 상태 체크
    checkAPIStatus();
    // 30초마다 체크
    setInterval(checkAPIStatus, 30000);
});